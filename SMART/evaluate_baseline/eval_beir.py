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


base = pyterrier_dr.SBertBiEncoder(model_name = f'/SMART/models/{args.model_name}')
index = pyterrier_dr.FlexIndex(f'/SMART/index/{args.dataset}-{args.model_name}.flex')
idx_pipeline = base >> index

topics = dataset.get_topics('text')

pipeline = bm25 % 100 >> base.query_encoder() >> index.scorer() 
pip = pipeline >> pta.fusion.PerQueryMaxMinScore()
bm25 = bm25 % 100 >> pta.fusion.PerQueryMaxMinScore()
linear = 0.5 * bm25 + 0.5 * pip

res = pt.Experiment(
    [linear],        
   topics,      
    dataset.get_qrels(),       
    eval_metrics=[nDCG@10],
    names=['l1']
)

print(f'Dataset: {args.dataset}')
print(f'Model: {args.model_name}')
print(res)
