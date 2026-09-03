import json
import logging
import os
import random

import hydra
import torch
import numpy as np
from tqdm import tqdm
from indxr import Indxr
from omegaconf import DictConfig

from model.gated_temp import GatedBiEncoder, GatedCrossEncoder
from model.utils import seed_everything

from ranx import Run, Qrels, compare
import subprocess

logger = logging.getLogger(__name__)


def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_jsonl(path):
    collection = {}
    with open(path) as f:
        # data = [json.loads(line) for line in tqdm(f, total=file_len(path))]
        for line in tqdm(f, total=file_len(path)):
            json_row = json.loads(line)
            collection[json_row['id']] = json_row['title']
    return collection

def get_bert_rerank(data, model, corpus, cfg):
    with open(cfg.general.label_file, 'r') as f:
        label_to_id = json.load(f)
    pred_id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    bert_run = {}
    accuracy = []
    pbar = tqdm(data, total=len(data))
    for d in pbar:
        
        with torch.no_grad():
            if cfg.general.gating_mode == 'top_k':
                domain = model.get_domain(d['text']).view(-1)
            if cfg.general.gating_mode == 'rand':
                domain = torch.rand(cfg.general.num_classes)
            # top_k = torch.topk(domain, 2).indices.cpu().detach().tolist()
            if 'cross' in model.model_name:
                domain_probs = domain.sigmoid().cpu().tolist()
                max_prob = domain.sigmoid().max().cpu().detach()

                if max_prob > 0.5:
                    domain = pred_id_to_label[domain.sigmoid().argmax().cpu().detach().tolist()]
                else:
                    domain = None
                
            else:
                domain_probs = domain.cpu().tolist()
                domain = pred_id_to_label[domain.argmax().cpu().detach().tolist()]
            
            if cfg.general.gating_mode == 'rand':
                domain = torch.rand(cfg.general.num_classes)
                domain_probs = domain.cpu().tolist()
                domain = pred_id_to_label[domain.argmax().cpu().detach().tolist()]
            
            # domain = [pred_id_to_label[k] for k in top_k]
            # model.set_active_adapters(domain, domain_probs, 'weighted')
            model.set_active_adapters(domain, domain_probs, cfg.general.gating_mode)
            if 'cross' not in model.model_name:
                q_embedding = model(d['text'])
                
            # q_embedding = model(d['text'])
                   
        accuracy.append(int(domain == cfg.dataset.name))
        bm25_docs = d['bm25_doc_ids']
        bm25_documents = [corpus[d_id] for d_id in bm25_docs]
        if 'cross' in model.model_name:
            with torch.no_grad():
                bert_scores = model(d['text'], bm25_documents)
        else:
            with torch.no_grad():
                d_embeddings = model(bm25_documents)
            bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}
        average_gate_accuracy = np.mean(accuracy)
        pbar.set_description('Accuracy {}'.format(round(average_gate_accuracy*100,2)))
    logging.info('Accuracy: {}'.format(round(average_gate_accuracy*100,2)))
    return bert_run


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_dir = cfg.dataset.data_dir
    runs_dir = cfg.dataset.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    logging_dir = cfg.logging.logging_dir
    logging_file = os.path.join(logging_dir, '11_test_moe.log')
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    corpus_path = os.path.join(cfg.dataset.data_dir, 'collection.jsonl')
    corpus = read_jsonl(corpus_path)
    
    if 'cross' in cfg.model.name:
        model_encoder = GatedCrossEncoder(
           cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.general.num_classes,
            cfg.model.max_tokens,
            cfg.model.normalize,
            cfg.model.device
        )
    else: 
        model_encoder = GatedBiEncoder(
            cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.general.num_classes,
            cfg.model.max_tokens,
            cfg.model.normalize,
            cfg.model.pooling_mode,
            cfg.model.device
        )
    model_encoder.load(os.path.join(cfg.general.all_model_dir,f'{cfg.model.name}.pt'))
    with open(cfg.general.label_file, 'r') as f:
        domain_to_id = json.load(f)
    domains = list(domain_to_id.keys())
    adapter_lists = [
        os.path.join(cfg.general.general_model_dir, domain , f'{cfg.model.name}_adapter')
        for domain in domains
    ]
    model_encoder.load_adapters(adapter_lists)
    model_encoder.load_adapter(os.path.join(cfg.general.all_model_dir, f'{cfg.model.name}_cls_adapter'))
    model_encoder.load_cls(os.path.join(cfg.general.all_model_dir, f'{cfg.model.name}_cls_head.pt'))
    
    split = 'val'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')

    bert_run = get_bert_rerank(query_data, model_encoder, corpus, cfg)

    ranx_run = Run(bert_run, 'Neural')
    os.makedirs(os.path.join(cfg.dataset.runs_dir, split), exist_ok=True)
    ranx_run.save(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_{cfg.general.gating_mode}.lz4'))

    split = 'test'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    bert_run = get_bert_rerank(query_data, model_encoder, corpus, cfg)

    ranx_run = Run(bert_run, 'Neural')
    os.makedirs(os.path.join(cfg.dataset.runs_dir, split), exist_ok=True)
    ranx_run.save(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}_moe_{cfg.general.gating_mode}.lz4'))
    ranx_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))
    

    models = [ranx_run]
    evaluation_report = compare(
        ranx_qrels,
        models,
        ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3']
    )

    print(evaluation_report)
    logging.info(f"Results:\n{evaluation_report}")

if __name__ == '__main__':
    main()