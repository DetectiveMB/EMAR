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

from sklearn.metrics import classification_report, confusion_matrix
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

def get_bert_rerank(data, model, cfg):
    with open(cfg.general.label_file, 'r') as f:
        label_to_id = json.load(f)
    pred_id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    accuracy = []
    classified_domain = []
    pbar = tqdm(data, total=len(data))
    for d in pbar:
        
        with torch.no_grad():
            domain = model.get_domain(d['text']).view(-1)
            if 'cross' in model.model_name:
                domain_probs = domain.sigmoid().cpu().tolist()
                max_prob = domain.sigmoid().max().cpu().detach()

                if max_prob > 0.5:
                    domain = pred_id_to_label[domain.sigmoid().argmax().cpu().detach().tolist()]
                else:
                    domain = 'None'
                
            else:
                domain_probs = domain.cpu().tolist()
                domain = pred_id_to_label[domain.argmax().cpu().detach().tolist()]
            
            if cfg.general.gating_mode == 'rand':
                domain = torch.rand(cfg.general.num_classes)
                domain_probs = domain.cpu().tolist()
                domain = pred_id_to_label[domain.argmax().cpu().detach().tolist()]

        # if domain != cfg.dataset.name:
            # print(domain, ":", d['text'])
        classified_domain.append(domain)                   
        accuracy.append(int(domain == cfg.dataset.name))
        average_gate_accuracy = np.mean(accuracy)
        # pbar.set_description('Accuracy {}'.format(round(average_gate_accuracy*100,2)))
    logging.info('Accuracy: {}'.format(round(average_gate_accuracy*100,2)))
    return classified_domain


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data_dir = cfg.dataset.data_dir
    runs_dir = cfg.dataset.runs_dir
    os.makedirs(runs_dir, exist_ok=True)

    logging_dir = cfg.logging.logging_dir
    logging_file = os.path.join(logging_dir, '13_analysis.log')
    logging.basicConfig(
        filename=logging_file,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO
    )

    seed_everything(cfg.general.seed)
    
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
    
    model_encoder.load_adapter(os.path.join(cfg.general.all_model_dir, f'{cfg.model.name}_cls_adapter'))
    model_encoder.load_cls(os.path.join(cfg.general.all_model_dir, f'{cfg.model.name}_cls_head.pt'))

    with open(cfg.general.label_file, 'r') as f:
        domain_to_id = json.load(f)
    domains = list(domain_to_id.keys())

    split = 'val'
    # query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')

    # bert_acc = get_bert_rerank(query_data, model_encoder, cfg)
    # y_true = [cfg.dataset.name for _ in bert_acc]

    
    split = 'test'
    query_data = Indxr(os.path.join(data_dir, split, 'queries.jsonl'), key_id='id')
    bert_acc = get_bert_rerank(query_data, model_encoder, cfg)
    y_true = [cfg.dataset.name for _ in bert_acc]
    print(confusion_matrix(y_true, bert_acc, labels=domains))
    
    
if __name__ == '__main__':
    main()