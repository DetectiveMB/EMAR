import json
import random
import torch

from indxr import Indxr
from torch.utils.data import Dataset
from os.path import join
from ranx import Run
from tqdm import tqdm

class MultiDomainDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        super(MultiDomainDataset, self).__init__()  
        self.data_dir = data_dir
        self.split = split

        self.split_dir = join(data_dir, split)

        self.corpus = Indxr(join(self.data_dir, 'collection.jsonl'), key_id='id')
        self.queries = Indxr(join(self.split_dir, 'queries.jsonl'), key_id='id')

    
    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['title'] if 'title' in query else query['text']
        rel_doc_id = random.choice(query['rel_doc_ids'])
        pos_doc_text = self.corpus.get(rel_doc_id)['title']

        retrieved_doc_ids = [q for q in query["bm25_doc_ids"] if q not in query['rel_doc_ids']]
        if retrieved_doc_ids:
            neg_doc_id = random.choice(retrieved_doc_ids)
            neg_doc_text = self.corpus.get(neg_doc_id)['title']
        else:
            neg_doc_text = '[SEP]'

        domain = query.get('domain', '')

        return {'query': query_text, 'pos_doc': pos_doc_text, 'neg_doc': neg_doc_text, 'domain': domain}

    def __len__(self):
        return len(self.queries)


class GateDomainDataset(Dataset):
    def __init__(self, data_dir: str, label_file: str, split: str):
        super(GateDomainDataset, self).__init__()  
        self.data_dir = data_dir
        self.split = split
        with open(label_file, 'r') as f:
            self.label_to_id = json.load(f)
        self.data = self.load_jsonl(join(self.data_dir, self.split, 'queries.jsonl'), text_field='text')    

    
    def __getitem__(self, idx):
        datapoint = self.data[idx]
        return {
            'text': datapoint['text'], 
            'domain': self.label_to_id[datapoint['domain']]
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_jsonl(file, text_field):
        file_row = []
        with open(file, 'r') as f:
            for row in tqdm(f):
                json_row = json.loads(row)
                file_row.append({
                    'id': json_row['id'], 
                    'domain': json_row['domain'], 
                    'text': json_row[text_field]
                })
                
        return file_row
    

class MultiDomainDatasetDistillation(Dataset):
    def __init__(self, data_dir: str, split: str, embedding_dir: str, runs_dir: str, model_name: str, num_negatives=1):
        super(MultiDomainDatasetDistillation, self).__init__()  
        self.data_dir = data_dir
        self.split = split
        self.model_name = model_name

        self.split_dir = join(data_dir, split)
        self.embedding_dir = embedding_dir
        self.runs_dir = runs_dir

        self.corpus = Indxr(join(self.data_dir, 'collection.jsonl'), key_id='id')
        self.queries = Indxr(join(self.split_dir, 'queries.jsonl'), key_id='id')
        self.run = Run.from_file(join(self.runs_dir, f'{self.split}/{self.model_name}.lz4'))

        self.num_negatives = num_negatives

    
    def __getitem__(self, idx):
        query = self.queries[idx]
        query_text = query['text']
        rel_doc_id = random.choice(query['rel_doc_ids'])
        pos_doc_text = self.corpus.get(rel_doc_id)['title'] 

        
        retrieved_doc_ids = [q for q in query["bm25_doc_ids"] if q not in query['rel_doc_ids']]
        if retrieved_doc_ids:
            # if len(retrieved_doc_ids) < self.num_negatives:
            #     neg_doc_ids = random.choices(retrieved_doc_ids, k=self.num_negatives) # with replacement, better than padding
            # else:
            #     neg_doc_ids = random.sample(retrieved_doc_ids, k=self.num_negatives)
            # neg_doc_texts = [self.corpus.get(neg_doc_id)['title'] for neg_doc_id in neg_doc_ids]
            neg_doc_id = random.choice(retrieved_doc_ids)
            neg_doc_text = self.corpus.get(neg_doc_id)['title'] 
        
        # get scores from bm25
        try:
            p_score = self.run[query['id']][rel_doc_id]
        except KeyError:
            p_score = 1.

        # n_scores = [self.run[query['id']].get(neg_doc_id, 0.) for neg_doc_id in neg_doc_ids]
        n_score = self.run[query['id']][neg_doc_id]

        return {
            'query': query_text,
            'pos_doc': pos_doc_text,
            'neg_doc': neg_doc_text,
            'P_scores': p_score,
            'N_scores': n_score,
        }

    def __len__(self):
        return len(self.queries)
    

def collate_fn(batch):
    return {
        'query': [item['query'] for item in batch],
        'pos_doc': [item['pos_doc'] for item in batch],
        'neg_doc': [item['neg_doc'] for item in batch],
        'P_scores': torch.tensor([item['P_scores'] for item in batch]),
        'N_scores': torch.tensor([item['N_scores'] for item in batch]),
    }
