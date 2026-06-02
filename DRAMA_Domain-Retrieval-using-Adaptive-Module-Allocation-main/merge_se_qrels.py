import os
import json

data_folder = './SE-PQA_extra'
output_folder = os.path.join(data_folder, 'all')
os.makedirs(os.path.join(output_folder, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)

merged_val_qrels = {}
merged_test_qrels = {}

# Iterate through each community folder
for comm in os.listdir(data_folder):
    comm_path = os.path.join(data_folder, comm)
    if os.path.isdir(comm_path) and comm != 'all':
        for split in ['val', 'test']:
            qrels_path = os.path.join(comm_path, split, 'qrels.json')
            if os.path.exists(qrels_path):
                with open(qrels_path, 'r') as f:
                    qrels = json.load(f)
                    if split == 'val':
                        merged_val_qrels.update(qrels)
                    elif split == 'test':
                        merged_test_qrels.update(qrels)

# Save the merged qrels to the respective folders in the all folder
merged_val_qrels_path = os.path.join(output_folder, 'val', 'qrels.json')
with open(merged_val_qrels_path, 'w') as f:
    json.dump(merged_val_qrels, f, indent=2)

merged_test_qrels_path = os.path.join(output_folder, 'test', 'qrels.json')
with open(merged_test_qrels_path, 'w') as f:
    json.dump(merged_test_qrels, f, indent=2)

print(f"Merged val qrels saved to {merged_val_qrels_path}")
print(f"Merged test qrels saved to {merged_test_qrels_path}")