import json
import logging
import os

import hydra

from indxr import Indxr
from omegaconf import DictConfig
from torch import no_grad, save, zeros
from tqdm import tqdm

from model.utils import seed_everything
from model.model import BiEncoder


logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = cfg.dataset.data_dir
    os.makedirs(cfg.dataset.embedding_dir, exist_ok=True)
    logging_dir = cfg.logging.logging_dir
    os.makedirs(logging_dir, exist_ok=True)
    model_dir = cfg.dataset.model_dir
    os.makedirs(model_dir, exist_ok=True)
    logging_file = os.path.join(logging_dir, '5_query_embedding.log')
    logging.basicConfig(filename=logging_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    
    seed_everything(cfg.general.seed)
    
    split = 'train'
    queries = Indxr(os.path.join(data_dir, f'{split}/queries.jsonl'), key_id='id')
    queries = sorted(queries, key=lambda k: len(k.get("text", "")), reverse=True)
    
    split = 'val'
    val_queries = Indxr(os.path.join(data_dir, f'{split}/queries.jsonl'), key_id='id')
    val_queries = sorted(val_queries, key=lambda k: len(k.get("text", "")), reverse=True)
    
    split = 'test'
    test_queries = Indxr(os.path.join(data_dir, f'{split}/queries.jsonl'), key_id='id')
    test_queries = sorted(test_queries, key=lambda k: len(k.get("text", "")), reverse=True)
    
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

    index = 0
    texts = []
    id_to_index = {}
    embedding_matrix = zeros(len(queries) + len(val_queries) + len(test_queries), cfg.model.embedding_size).float()

    for query in tqdm(queries):
        id_to_index[query['id']] = index
        index += 1
        texts.append(query.get('text',''))
        if len(texts) == cfg.dataset.batch_size:
            with no_grad():
                embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
        texts = []

    for query in tqdm(val_queries):
        id_to_index[query['id']] = index
        index += 1
        texts.append(query.get('text',''))
        if len(texts) == cfg.dataset.batch_size:
            with no_grad():
                embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()

    for query in tqdm(test_queries):
        id_to_index[query['id']] = index
        index += 1
        texts.append(query.get('text',''))
        if len(texts) == cfg.dataset.batch_size:
            with no_grad():
                embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            embedding_matrix[index - len(texts) : index] = model.query_encoder(texts).cpu()

    save(embedding_matrix, os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}_query.pt'))

    with open(os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}_query.json'), 'w') as f:
        json.dump(id_to_index, f)

if __name__ == '__main__':
    main()