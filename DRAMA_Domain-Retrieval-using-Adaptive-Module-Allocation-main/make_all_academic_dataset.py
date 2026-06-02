import json
import os

def main():
    os.makedirs('dataset/all', exist_ok=True)
    os.makedirs('dataset/all/train', exist_ok=True)
    os.makedirs('dataset/all/val', exist_ok=True)
    os.makedirs('dataset/all/test', exist_ok=True)

    all_collection = []
    total = 0

    for current_dataset in ['physics', 'psychology', 'political_science', 'computer_science']:
        with open(f'dataset/{current_dataset}/collection.jsonl', 'r') as f:
            current_dataset_collection = [json.loads(line) for line in f]
        for c in current_dataset_collection:
            c['domain'] = current_dataset
        total += len(current_dataset_collection)
        all_collection.extend(current_dataset_collection)

    assert len(all_collection) == total
    with open('dataset/all/collection.jsonl', 'w') as f:
        f.write("\n".join([json.dumps(y) for y in all_collection]))

    print('Saved the complete collection')
    
    for split in ['train', 'val', 'test']:
        all_queries = []
        total = 0

        for current_dataset in ['physics', 'psychology', 'political_science', 'computer_science']:
            with open(f'dataset/{current_dataset}/{split}/queries.jsonl', 'r') as f:
                current_dataset_queries = [json.loads(line) for line in f]
            for c in current_dataset_queries:
                c['domain'] = current_dataset
            total += len(current_dataset_queries)
            all_queries.extend(current_dataset_queries)

        assert len(all_queries) == total
        with open(f'dataset/all/{split}/queries.jsonl', 'w') as f:
            f.write("\n".join([json.dumps(y) for y in all_queries]))
            f.write("\n")
        print(f'Saved the {split} queries')


    with open('dataset/all/label_to_id.json', 'w') as f:
        json.dump(dict((v, k) for k, v in enumerate(['physics', 'psychology', 'political_science', 'computer_science'])), f, indent=2)
    
if __name__ == '__main__':
    main()