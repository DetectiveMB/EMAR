import pandas as pd
import pyterrier_dr 
import pyterrier
import argparse
import json

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset', type=str, default='trec-covid')
parser.add_argument('--df_name', type=str, default='CS') # E5, RetroMAE
parser.add_argument('--model_name', type=str, default='msmarco-bert-base-dot-v5')
#parser.add_argument('--layers', type=list, default=[7, 3])
args = parser.parse_args()

corpus = []
current_dataset_collection = []

with open(f'/SMART/data/{args.df_name}/collection.jsonl', 'r') as f:
    for i, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
            current_dataset_collection.append(obj)

        except json.JSONDecodeError:
            print(f"Skipping invalid JSON at line {i}")
            continue
    
    for doc in current_dataset_collection:
        corpus.append({'id':doc['id'], 'text': doc['title']+' '+doc['text']})

    df = pd.DataFrame(corpus)

    df.to_csv(f'/SMART/data/{args.df_name}/collection.csv', index=False)


corpus = pd.read_csv(f'/SMART/data/{args.df_name}/collection.csv', engine = 'python', on_bad_lines='skip')
print(len(corpus))
corpus = corpus.dropna(subset=['text'])
corpus = corpus[corpus['text'].str.strip() != '']

def docs_iter(df):
    for _, row in df.iterrows():
        yield {
                "docno": str(row["id"]),
                "text": row["text"]
        }

model = pyterrier_dr.SBertBiEncoder(model_name = '/SMART/models/' + args.model_name)
index = pyterrier_dr.FlexIndex(f'/SMART/index/{args.df_name}-{args.model_name}.flex')
idx_pipeline = model >> index

idx_pipeline.index(docs_iter(corpus))
