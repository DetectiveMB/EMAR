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
    logging_file = os.path.join(logging_dir, '2_create_embedding.log')
    logging.basicConfig(filename=logging_file,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO
                        )
    
    seed_everything(cfg.general.seed)

    corpus = Indxr(os.path.join(data_dir, 'collection.jsonl'), key_id='id')
    corpus = sorted(corpus, key=lambda k: len(k.get("title", "") + k.get("text", "")), reverse=True)
    
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
    embedding_matrix = zeros(len(corpus), cfg.model.embedding_size).float()

    for doc in tqdm(corpus):
        id_to_index[doc['id']] = index
        index += 1
        texts.append(doc.get('title','')) # + '. ' + doc['text'])
        if len(texts) == cfg.dataset.batch_size:
            with no_grad():
                embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()
            texts = []
    if texts:
        with no_grad():
            embedding_matrix[index - len(texts) : index] = model.doc_encoder(texts).cpu()

    save(embedding_matrix, os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}.pt'))

    with open(os.path.join(cfg.dataset.embedding_dir, f'{cfg.model.name}.json'), 'w') as f:
        json.dump(id_to_index, f)

if __name__ == '__main__':
    main()