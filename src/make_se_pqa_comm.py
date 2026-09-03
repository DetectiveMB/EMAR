import os
import json
import subprocess
from tqdm import tqdm
from retriv import SparseRetriever
from ranx import evaluate

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def read_jsonl(path):
    with open(path) as f:
        data = [json.loads(line) for line in tqdm(f, total=file_len(path))]
    return data

def write_jsonl(path, data):
    with open(path, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

data_folder = './SE-PQA'
output_datafolder = 'SE-PQA_comm'
MIN_DOC = 50_000

comms = [
    'writers',
    'workplace',
    'woodworking',
    'vegetarianism',
    'travel',
    'sustainability',
    'sports',
    'sound',
    'skeptics',
    'scifi',
    'rpg',
    'politics',
    'philosophy',
    'pets',
    'parenting',
    'outdoors',
    'opensource',
    'musicfans',
    'music',
    'movies',
    'money',
    'martialarts',
    'literature',
    'linguistics',
    'lifehacks',
    'law',
    'judaism',
    'islam',
    'interpersonal',
    'hsm',
    'history',
    'hinduism',
    'hermeneutics',
    'health',
    'genealogy',
    'gardening',
    'gaming',
    'freelancing',
    'fitness',
    'expatriates',
    'english',
    'diy',
    'cooking',
    'christianity',
    'buddhism',
    'boardgames',
    'bicycles',
    'apple',
    'anime',
    'academia',
]

with open(os.path.join(data_folder, 'answer_collection.json'), 'r') as f:
    collection = json.load(f)

train_data = read_jsonl(os.path.join(data_folder, 'train', 'data.jsonl'))
val_data = read_jsonl(os.path.join(data_folder, 'val', 'data.jsonl'))
test_data = read_jsonl(os.path.join(data_folder, 'test', 'data.jsonl'))
    
good_comm = []
for comm in comms:
    comm_coll = []
    for ans_id, ans in collection.items():
        if comm == ans_id.split('_')[0]:
            comm_coll.append({
                'id': ans_id,
                'title': str(ans),
                'text': str(ans)
            })
    
    comm_train_data = [
        {
            'id': doc['id'],
            'title': str(doc['title']),
            'text': str(doc['text']),
            'rel_doc_ids': doc['rel_ids'],
            'best_answer': doc['best_answer']
        }
        for doc in train_data
        if comm == doc['id'].split('_')[0]
    ]
    
    comm_val_data = [
        {
            'id': doc['id'],
            'title': str(doc['title']),
            'text': str(doc['text']),
            'rel_doc_ids': doc['rel_ids'],
            'best_answer': doc['best_answer']
        }
        for doc in val_data
        if comm == doc['id'].split('_')[0]
    ]
    
    comm_test_data = [
        {
            'id': doc['id'],
            'title': str(doc['title']),
            'text': str(doc['text']),
            'rel_doc_ids': doc['rel_ids'],
            'best_answer': doc['best_answer']
        }
        for doc in test_data
        if comm == doc['id'].split('_')[0]
    ]
    if len(comm_train_data) > MIN_DOC:
        good_comm.append(comm)
        print(comm, len(comm_train_data))
        os.makedirs(os.path.join(output_datafolder, comm), exist_ok=True)
        os.makedirs(os.path.join(output_datafolder, comm, 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_datafolder, comm, 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_datafolder, comm, 'test'), exist_ok=True)
        
        write_jsonl(os.path.join(output_datafolder, comm, 'train', 'data.jsonl'), comm_train_data)
        write_jsonl(os.path.join(output_datafolder, comm, 'collection.jsonl'), comm_coll)
        write_jsonl(os.path.join(output_datafolder, comm, 'val', 'data.jsonl'), comm_val_data)
        write_jsonl(os.path.join(output_datafolder, comm, 'test', 'data.jsonl'), comm_test_data)
        

print(good_comm)

all_coll = []
all_train = []
all_val = []
all_test = []

for comm in tqdm(good_comm):
    sr = SparseRetriever(
        index_name=comm,
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
        hyperparams=dict(b=1, k1=1.75)
    )
    
    collection_jsonl = read_jsonl(os.path.join(output_datafolder, comm, 'collection.jsonl'))
    sr = sr.index(
        collection_jsonl,
        show_progress=True,         # Default value
        callback=lambda doc: {      # Callback defaults to None.
            "id": doc["id"],
            "text": str(doc["text"]),
            "domain": 'se_' + comm
        }   
    )
    all_coll.extend(collection_jsonl)
    
    train_data = read_jsonl(os.path.join(output_datafolder, comm, 'train', 'data.jsonl'))
    val_data = read_jsonl(os.path.join(output_datafolder, comm, 'val', 'data.jsonl'))
    test_data = read_jsonl(os.path.join(output_datafolder, comm, 'test', 'data.jsonl'))

    train_search_res = sr.bsearch(train_data)
    val_search_res = sr.bsearch(val_data)
    test_search_res = sr.bsearch(test_data)

    train_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['text'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': list(train_search_res[d['id']].keys()),
            'bm25_doc_scores': [x.item() for x in list(train_search_res[d['id']].values())],
            'domain': 'se_' + comm
        }
        for d in train_data
    ]

    val_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['text'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': list(val_search_res[d['id']].keys()),
            'bm25_doc_scores': [x.item() for x in list(val_search_res[d['id']].values())],
            'domain': 'se_' + comm
        }
        for d in val_data
    ]

    test_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['text'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': list(test_search_res[d['id']].keys()),
            'bm25_doc_scores': [x.item() for x in list(test_search_res[d['id']].values())],
            'domain': 'se_' + comm
        }
        for d in test_data
    ]
    write_jsonl(os.path.join(output_datafolder, comm, 'train', 'queries.jsonl'), train_queries)
    write_jsonl(os.path.join(output_datafolder, comm, 'val', 'queries.jsonl'), val_queries)
    write_jsonl(os.path.join(output_datafolder, comm, 'test', 'queries.jsonl'), test_queries)
    all_train.extend(train_queries)
    all_val.extend(val_queries)
    all_test.extend(test_queries)

    sr.delete(comm)
    qrels = {
        d['id']: {
            doc_id: 1
            for doc_id in d['rel_doc_ids']
        }
        for d in test_data
    }
    print(comm, evaluate(qrels, test_search_res, ['recall@100', 'precision@1']))

    train_qrels = {
        d['id']: {
            doc_id: 1
            for doc_id in d['rel_doc_ids']
        }
        for d in train_data
    }
    with open(os.path.join(output_datafolder, comm, 'train', 'qrels.json'), 'w') as f:
        json.dump(train_qrels, f)
    val_qrels = {
        d['id']: {
            doc_id: 1
            for doc_id in d['rel_doc_ids']
        }
        for d in val_data
    }
    with open(os.path.join(output_datafolder, comm, 'val', 'qrels.json'), 'w') as f:
        json.dump(val_qrels, f)
    test_qrels = {
        d['id']: {
            doc_id: 1
            for doc_id in d['rel_doc_ids']
        }
        for d in test_data
    }
    with open(os.path.join(output_datafolder, comm, 'test', 'qrels.json'), 'w') as f:
        json.dump(test_qrels, f)


os.makedirs(os.path.join(output_datafolder, 'all'), exist_ok=True)
os.makedirs(os.path.join(output_datafolder, 'all', 'train'), exist_ok=True)
os.makedirs(os.path.join(output_datafolder, 'all', 'val'), exist_ok=True)
os.makedirs(os.path.join(output_datafolder, 'all', 'test'), exist_ok=True)
        

write_jsonl(os.path.join(output_datafolder, 'all', 'collection.jsonl'), all_coll)
write_jsonl(os.path.join(output_datafolder, 'all', 'train', 'queries.jsonl'), all_train)
write_jsonl(os.path.join(output_datafolder, 'all', 'val', 'queries.jsonl'), all_val)
write_jsonl(os.path.join(output_datafolder, 'all', 'test', 'queries.jsonl'), all_test)

with open(os.path.join(output_datafolder, 'all', 'label_to_id.json'), 'w') as f:
    json.dump(dict(('se_'+v, k) for k, v in enumerate(good_comm)), f, indent=2)
"""
for comm in good_comm:
    train_data = read_jsonl(os.path.join(output_datafolder, comm, 'train', 'queries.jsonl'))
    val_data = read_jsonl(os.path.join(output_datafolder, comm, 'val', 'queries.jsonl'))
    test_data = read_jsonl(os.path.join(output_datafolder, comm, 'test', 'queries.jsonl'))

    train_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['title'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': d['bm25_doc_ids'],
            'bm25_doc_scores': d['bm25_doc_scores']
        }
        for d in train_data
    ]

    val_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['title'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': d['bm25_doc_ids'],
            'bm25_doc_scores': d['bm25_doc_scores']
        }
        for d in val_data
    ]

    test_queries = [
        {
            'id': d['id'],
            'title': d['title'],
            'text': d['title'],
            'rel_doc_ids': d['rel_doc_ids'],
            'bm25_doc_ids': d['bm25_doc_ids'],
            'bm25_doc_scores': d['bm25_doc_scores']
        }
        for d in test_data
    ]
    write_jsonl(os.path.join(output_datafolder, comm, 'train', 'queries.jsonl'), train_queries)
    write_jsonl(os.path.join(output_datafolder, comm, 'val', 'queries.jsonl'), val_queries)
    write_jsonl(os.path.join(output_datafolder, comm, 'test', 'queries.jsonl'), test_queries)
"""
