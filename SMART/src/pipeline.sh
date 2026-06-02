MODEL=bert
DATASETS="physics political_science computer_science psychology"

for DATASET in $DATASETS
do 
    python3 1_train.py dataset=$DATASET model=$MODEL dataset.max_epoch=40 dataset.batch_size=128
    echo $DATASET
    python3 2_create_embedding.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 3_test.py dataset=$DATASET model=$MODEL
    python3 3_test_full_rank.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL
done

MODEL=distilbert
for DATASET in $DATASETS
do 
    python3 1_train.py dataset=$DATASET model=$MODEL dataset.max_epoch=40 
    echo $DATASET
    python3 2_create_embedding.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 3_test.py dataset=$DATASET model=$MODEL
    python3 3_test_full_rank.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL
done

#all training
MODEL=distilbert
python3 1_train.py dataset=all model=$MODEL dataset.max_epoch=40


DATASETS="physics psychology political_science computer_science"

for DATASET in $DATASETS
do 
    python3 2_create_embedding.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
    python3 3_test.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
    python3 3_test_full_rank.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
done

MODEL=distilbert
DATASETS="physics psychology political_science computer_science"

for DATASET in $DATASETS
do 
    echo $DATASET
    python3 6_train_moe.py dataset=$DATASET model=$MODEL dataset.lr=1e-4 dataset.max_epoch=40 general.reduction_factor=4 general.patience=41 dataset.batch_size=128
    python3 7_create_embedding_moe.py dataset=$DATASET model=$MODEL
    python3 8_test_moe.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 9_fuse_moe.py dataset=$DATASET model=$MODEL
done


python3 10_train_gate.py dataset=all dataset.lr=1e-4 model=$MODEL dataset.max_epoch=10

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 13_classification_analysis.py dataset=$DATASET model=$MODEL general.gating_mode='top_k'
done

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL general.gating_mode='rand'
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL general.gating_mode='rand'
    python3 13_classification_analysis.py dataset=$DATASET model=$MODEL general.gating_mode='rand'
done


MODEL=cross_tinybert
DATASETS="physics political_science computer_science psychology"
DEVICE='cuda'
for DATASET in $DATASETS
do 
    # python3 1_train.py dataset=$DATASET model=$MODEL model.device=$DEVICE dataset.max_epoch=40 dataset.batch_size=128
    echo $DATASET
    python3 3_test.py dataset=$DATASET model=$MODEL model.device=$DEVICE
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL model.device=$DEVICE
done

python3 1_train.py dataset=all model=$MODEL model.device=$DEVICE dataset.max_epoch=40

DATASETS="physics political_science computer_science psychology"
for DATASET in $DATASETS
do 
    python3 3_test.py dataset=$DATASET model=$MODEL model.device=$DEVICE dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL model.device=$DEVICE dataset.model_dir=../models/all dataset.embedding_dir=../embeddings/all/$DATASET dataset.runs_dir=../runs/all/$DATASET logging.logging_dir=../logs/all/$DATASET/$MODEL
done

for DATASET in $DATASETS
do 
    echo $DATASET
    python3 6_train_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE dataset.lr=1e-4 dataset.max_epoch=40 general.reduction_factor=4 general.patience=41 dataset.batch_size=128
    python3 8_test_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE
    echo $DATASET
    python3 9_fuse_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE
done

python3 10_train_gate.py dataset=all dataset.lr=1e-5 model=$MODEL model.device=$DEVICE dataset.max_epoch=10

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE
    echo $DATASET
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE
    echo $DATASET
    python3 13_classification_analysis.py dataset=$DATASET model=$MODEL model.device=$DEVICE general.gating_mode='top_k'
done

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE general.gating_mode='rand' 
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL model.device=$DEVICE general.gating_mode='rand'
    # python3 13_classification_analysis.py dataset=$DATASET model=$MODEL model.device=$DEVICE general.gating_mode='rand'
done

for DATASET in $DATASETS
do 
    echo $DATASET
    python 14_test.py dataset=$DATASET model=$MODEL model.device=$DEVICE
done

MODEL=all_minilm
DATASETS="se_scifi se_gaming se_english se_apple"

for DATASET in $DATASETS
do 
    python3 1_train.py dataset=$DATASET model=$MODEL dataset.max_epoch=40 dataset.batch_size=128
    echo $DATASET
    python3 2_create_embedding.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 3_test.py dataset=$DATASET model=$MODEL
    python3 3_test_full_rank.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL
done

MODEL=cross_minilm
DATASETS="se_scifi se_gaming se_english se_apple"

for DATASET in $DATASETS
do 
    echo $DATASET
    python3 1_train.py dataset=$DATASET model=$MODEL dataset.max_epoch=40
    echo $DATASET
    #python3 2_create_embedding.py dataset=$DATASET model=$MODEL
    echo $DATASET
    python3 3_test.py dataset=$DATASET model=$MODEL
    echo "${DATASET}"
    echo "${MODEL}"
    python3 4_fuse.py dataset=$DATASET model=$MODEL
done

MODEL=all_minilm
python3 1_train.py dataset=se_all model=$MODEL dataset.max_epoch=40
# DATASETS="se_workplace se_travel se_scifi se_rpg se_money se_judaism se_gaming se_english se_diy se_cooking se_apple se_academia"
DATASETS="se_scifi se_gaming se_english se_apple"

for DATASET in $DATASETS
do 
    # python3 2_create_embedding.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/se_all dataset.embedding_dir=../embeddings/se_all/$DATASET dataset.runs_dir=../runs/se_all/$DATASET logging.logging_dir=../logs/se_all/$DATASET/$MODEL
    python3 3_test.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/se_all dataset.embedding_dir=../embeddings/se_all/$DATASET dataset.runs_dir=../runs/se_all/$DATASET logging.logging_dir=../logs/se_all/$DATASET/$MODEL
    # python3 3_test_full_rank.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/se_all dataset.embedding_dir=../embeddings/se_all/$DATASET dataset.runs_dir=../runs/se_all/$DATASET logging.logging_dir=../logs/se_all/$DATASET/$MODEL
    echo $DATASET
    python3 4_fuse.py dataset=$DATASET model=$MODEL dataset.model_dir=../models/se_all dataset.embedding_dir=../embeddings/se_all/$DATASET dataset.runs_dir=../runs/se_all/$DATASET logging.logging_dir=../logs/se_all/$DATASET/$MODEL
done

MODEL=cross_minilm
DATASETS="se_scifi se_gaming se_english se_apple"

for DATASET in $DATASETS
do 
    echo $DATASET
    python3 6_train_moe.py dataset=$DATASET model=$MODEL dataset.lr=1e-5 general.all_model_dir=../models/se_all general.patience=11 general.reduction_factor=4 dataset.max_epoch=10 dataset.batch_size=128
    # python3 7_create_embedding_moe.py dataset=$DATASET model=$MODEL general.all_model_dir=../models/se_all
    python3 8_test_moe.py dataset=$DATASET model=$MODEL general.all_model_dir=../models/se_all
    echo $DATASET
    python3 9_fuse_moe.py dataset=$DATASET model=$MODEL general.all_model_dir=../models/se_all
done

python3 10_train_gate.py dataset=se_all model=$MODEL dataset.lr=1e-4 general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all

MODEL=all_minilm
DATASETS="se_scifi se_gaming se_english se_apple"

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
    echo $DATASET
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
    echo $DATASET
    python3 13_classification_analysis.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
done

for DATASET in $DATASETS
do 
    python3 11_test_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='rand'
    echo $DATASET
    python3 12_fuse_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='rand'
done


python3 11_test_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
echo $DATASET
python3 12_fuse_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
echo $DATASET
python3 13_classification_analysis.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='top_k'
python3 11_test_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='rand'
echo $DATASET
python3 12_fuse_moe.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='rand'
echo $DATASET
python3 13_classification_analysis.py dataset=$DATASET model=$MODEL general.label_file='../SE-PQA_comm/all/label_to_id.json' general.all_model_dir=../models/se_all general.gating_mode='rand'


MODEL=distilbert
DATASETS="physics political_science computer_science psychology"
for DATASET in $DATASETS
do 
    echo $DATASET
    python 14_test.py dataset=$DATASET model=$MODEL
done




MODEL=all_minilm
DATASETS="se_scifi se_gaming se_english se_apple"
for DATASET in $DATASETS
do 
    echo $DATASET
    python 14_test.py dataset=$DATASET model=$MODEL general.all_runs_dir='../runs/se_all'
done

