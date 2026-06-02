#!/bin/bash
#SBATCH -J /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_CS
#SBATCH -o /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_CS.%J.txt
#SBATCH -e /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_CS.%J.txt
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=4         
#SBATCH --time=20:20:00              
#SBATCH --account=IscrC_MOSAICIR      
#SBATCH --gres=gpu:1 
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal #normal boost_qos_dbg
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL

#export IR_DATASETS_HOME=/leonardo_work/IscrC_MiKroLLM/Braga/.ir_datasets
#LOCAL_SCRATCH=/leonardo_work/IscrC_MiKroLLM/Braga/local/
#LOCAL_HOME=/leonardo_work/IscrC_MiKroLLM/Braga/home/

#export HOME=$LOCAL_HOME

#export TRANSFORMERS_CACHE=$LOCAL_SCRATCH 
# export TORCH_CACHE=$LOCAL_SCRATCH 
# export SENTENCE_TRANSFORMERS_HOME=$LOCAL_SCRATCH 
# export MPLCONFIGDIR=$LOCAL_SCRATCH

# export HF_HOME=$LOCAL_SCRATCH
# export HUGGINGFACE_HUB_CACHE=$LOCAL_SCRATCH
# export HF_DATASETS_CACHE=$LOCAL_SCRATCH
# export HF_MODULES_CACHE=$LOCAL_SCRATCH
# export hf_cache_home=$LOCAL_SCRATCH
# export XDG_CACHE_HOME=$LOCAL_SCRATCH

module load profile/deeplrn
module load cineca-ai
source /leonardo_work/IscrC_MOSAICIR/Braga/tv_env/bin/activate

#for model in "Co-codenser" "E5" "RetroMAE" "RetroMAE_Distill"; do
#    for weight_ir in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "CS" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "CS" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "CS" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "CS" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "CS" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PH" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PH" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PH" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PH" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PH" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PS" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PS" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PS" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PS" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PS" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PSY" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PSY" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PSY" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PSY" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_baseline.py --dataset "PSY" --model_name "retromae_distil"


#    done
#done