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


queries = []
run_rows = []

with open(f"/SMART/data/{args.dataset}/queries.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)

        qid = str(obj["id"])

       
        queries.append({
            "qid": qid,
            "query": obj["text"]
        })

       
        doc_ids = obj["bm25_doc_ids"]
        scores = obj["bm25_doc_scores"]

        for rank, (docno, score) in enumerate(zip(doc_ids, scores), start=1):
            run_rows.append({
                "qid": qid,
                "docno": str(docno),
                "score": float(score),
                "rank": rank
            })

topics_df = pd.DataFrame(queries)
run_df = pd.DataFrame(run_rows)
topics_df["qid"] = topics_df["qid"].astype(str)
run_df["qid"] = run_df["qid"].astype(str)


with open(f"/data/{args.dataset}/qrels.json", "r") as f:
    qrels_dict = json.load(f)

qrels_rows = []

for qid, docs in qrels_dict.items():
    for docno, label in docs.items():
        qrels_rows.append({
            "qid": str(qid),
            "docno": str(docno),
            "label": int(label)
        })

qrels_df = pd.DataFrame(qrels_rows)

qrels_df["qid"] = qrels_df["qid"].astype(str)
qrels_df["docno"] = qrels_df["docno"].astype(str)

base = pyterrier_dr.SBertBiEncoder(model_name = f'/SMART/models/{args.model_name}')
index = pyterrier_dr.FlexIndex(f'/SMART/index/{args.dataset}-{args.model_name}.flex')
idx_pipeline = base >> index

candidate_set = pt.Transformer.from_df(
    run_df.merge(topics_df, on="qid")
)

pipeline = candidate_set % 1000 >> base.query_encoder() >> index.scorer()    # %1000 for Academic Search, %100 for SE-PQA and BEIR
pip = pipeline >> pta.fusion.PerQueryMaxMinScore()
bm25 = candidate_set % 1000 >> pta.fusion.PerQueryMaxMinScore()
linear = 0.5 * bm25 + 0.5 * pip
linear2 = 0.4 * bm25 + 0.6 * pip
linear3 = 0.3 * bm25 + 0.7 * pip

res = pt.Experiment(
    [linear, linear2, linear3],       
    topics_df,      
    qrels_df,       
    eval_metrics=[nDCG@10],
    names=['l1', 'l2', 'l3']
)

print(f'Dataset: {args.dataset}')
print(f'Model: {args.model_name}')
print(res)
