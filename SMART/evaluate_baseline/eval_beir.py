import pyterrier_dr
import pyterrier as pt
from pyterrier.measures import *
import pandas as pd
import json 
import pyterrier_alpha as pta

import argparse
parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='trec-covid')
parser.add_argument('--dataset', type=str, default='CS') # E5, RetroMAE
parser.add_argument('--model_name', type=str, default='msmarco-bert-base-dot-v5')

#parser.add_argument('--layers', type=list, default=[7, 3])
args = parser.parse_args()


if args.dataset=='nq/test':
    dataset = pt.get_dataset(f"irds:beir/nq")
else:
    dataset = pt.get_dataset(f"irds:beir/{args.dataset}")


index = pt.Artifact.from_hf(f'pyterrier/{args.dataset.split("/")[0]}.terrier')

bm25 = index.bm25()

#index_dir = f'/leonardo_scratch/fast/IscrC_MOSAICIR/SMART/index/{args.dataset}-bm25.flex'

# iter_indexer = pt.IterDictIndexer(
#         index_dir,
#         meta={"docno": 64, "text": 4096},
#         fields=["text"]
#     )

# index_ref = iter_indexer.index(dataset.get_corpus_iter())

#bm25 = pt.BatchRetrieve(index_dir, wmodel="BM25")

base = pyterrier_dr.SBertBiEncoder(model_name = f'/leonardo_scratch/fast/IscrC_MOSAICIR/SMART/models/{args.model_name}')
index = pyterrier_dr.FlexIndex(f'/leonardo_scratch/fast/IscrC_MOSAICIR/SMART/index/{args.dataset}-{args.model_name}.flex')
idx_pipeline = base >> index

import re
def clean_query(q):
        q = q.lower()
        q = re.sub(r"[\/:\?]", " ", q)   # also remove ?
        q = re.sub(r"[^\w\s]", " ", q)   # remove remaining punctuation safely
        q = re.sub(r"\s+", " ", q)
        q = re.sub(r"#", " ", q)
        return q.strip()

topics = dataset.get_topics('text')

topics = topics.rename(columns={'text': 'query'})

topics['query'] = topics['query'].apply(clean_query)

pipeline = bm25 % 100 >> base.query_encoder() >> index.scorer() 
pip = pipeline >> pta.fusion.PerQueryMaxMinScore()
bm25 = bm25 % 100 >> pta.fusion.PerQueryMaxMinScore()
linear = 0.5 * bm25 + 0.5 * pip

res = pt.Experiment(
    [linear],        # precomputed BM25 run
   topics,       # queries
    dataset.get_qrels(),        # relevance judgments
    eval_metrics=[nDCG@10],
    names=['l1']
)

print(f'Dataset: {args.dataset}')
print(f'Model: {args.model_name}')
print(res)