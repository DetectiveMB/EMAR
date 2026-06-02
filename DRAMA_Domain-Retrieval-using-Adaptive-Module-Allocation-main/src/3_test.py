import json
import logging
import os

import hydra
import torch
from tqdm import tqdm
from indxr import Indxr
from omegaconf import DictConfig

from model.model import BiEncoder, CrossEncoder
from model.utils import seed_everything

from ranx import Run, Qrels, compare

logger = logging.getLogger(__name__)


def get_bert_rerank(data, model, doc_embedding, id_to_index):
    bert_run = {}
    model.eval()
    for d in tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])

        bm25_docs = d['bm25_doc_ids']
        d_embeddings = doc_embedding[torch.tensor([int(id_to_index[x]) for x in bm25_docs])]
        bert_scores = torch.einsum('xy, ly -> x', d_embeddings, q_embedding)
        bert_run[d['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bm25_docs)}

    return bert_run

def get_cross_rerank(data, model, doc_collection):
    cross_run = {}
    model.eval()
    for d in tqdm(data, total=len(data)):
        bm25_docs = d['bm25_doc_ids']
        query_text = [d['text'] for _ in bm25_docs]
        doc_text = [d['title'] for d in doc_collection.mget(bm25_docs)]
        
        with torch.no_grad():
            cross_scores = model.classifier(query_text, doc_text)
        cross_run[d['id']] = {doc_id: cross_scores[i].item() for i, doc_id in enumerate(bm25_docs)}

    return cross_run

def get_full_bert_rank(data, model, doc_embedding, id_to_index, k=1000):
    bert_run = {}
    index_to_id = {ind: _id for _id, ind in id_to_index.items()}
    model.eval()
    for d in tqdm(data, total=len(data)):
        with torch.no_grad():
            q_embedding = model.query_encoder([d['text']])
        
        bert_scores = torch.einsum('xy, ly -> x', doc_embedding, q_embedding)
        index_sorted = torch.argsort(bert_scores, descending=True)
        top_k = index_sorted[:k]
        bert_ids = [index_to_id[int(_id)] for _id in top_k]
        bert_scores = bert_scores[top_k]
        bert_run[d['id']] = {doc_id: bert_scores[i].item() for i, doc_id in enumerate(bert_ids)}
        
        
    return bert_run

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_dir = cfg.dataset.data_dir
    model_dir = cfg.dataset.model_dir
    runs_dir = cfg.dataset.runs_dir
    

    logging_dir = cfg.logging.logging_dir
    os.makedirs(logging_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, '3_test.log')
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)

    if 'cross' in cfg.model.model_name:
        model = CrossEncoder(
            cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.model.max_tokens,
            cfg.model.device
        )
    else:
        model = BiEncoder(
            cfg.model.model_name,
            cfg.model.tokenizer_name,
            cfg.model.max_tokens,
            cfg.model.normalize,
            cfg.model.pooling_mode,
            cfg.model.device
        )
    model.load(os.path.join(model_dir,f'{cfg.model.name}.pt'))
    model.eval()
    
    if 'cross' in cfg.model.model_name:
        doc_collection = Indxr(os.path.join(cfg.dataset.data_dir, 'collection.jsonl'), key_id='id')
    else:
        doc_embedding = torch.load(os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}.pt')).to(cfg.model.device)
        with open(os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}.json'), 'r') as f:
            id_to_index = json.load(f)
    
    
    split = 'train'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    if 'cross' in cfg.model.model_name:
        bert_run = get_cross_rerank(query_data, model, doc_collection)
    else:
        bert_run = get_bert_rerank(query_data, model, doc_embedding, id_to_index)

    ranx_run = Run(bert_run, 'Neural')
    os.makedirs(os.path.join(cfg.dataset.runs_dir, split), exist_ok=True)
    ranx_run.save(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    """
    split = 'val'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    if 'cross' in cfg.model.model_name:
        bert_run = get_cross_rerank(query_data, model, doc_collection)
    else:
        bert_run = get_bert_rerank(query_data, model, doc_embedding, id_to_index)

    ranx_run = Run(bert_run, 'Neural')
    os.makedirs(os.path.join(cfg.dataset.runs_dir, split), exist_ok=True)
    ranx_run.save(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))

    split = 'test'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    if 'cross' in cfg.model.model_name:
        bert_run = get_cross_rerank(query_data, model, doc_collection)
    else:
        bert_run = get_bert_rerank(query_data, model, doc_embedding, id_to_index)

    ranx_run = Run(bert_run, 'Neural')
    os.makedirs(os.path.join(cfg.dataset.runs_dir, split), exist_ok=True)
    ranx_run.save(os.path.join(cfg.dataset.runs_dir, split ,f'{cfg.model.name}.lz4'))
    ranx_qrels = Qrels.from_file(os.path.join(data_dir, split, 'qrels.json'))
    
    models = [ranx_run]
    evaluation_report = compare(
        ranx_qrels,
        models,
        ['map@100', 'mrr@10', 'recall@100', 'ndcg@10', 'precision@1', 'ndcg@3']
    )

    print(evaluation_report)
    logging.info(f"Results:\n{evaluation_report}")
    """

if __name__ == '__main__':
    main()