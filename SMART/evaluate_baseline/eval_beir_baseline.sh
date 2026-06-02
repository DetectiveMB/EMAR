#!/bin/bash
#SBATCH -J /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_baseline_beir
#SBATCH -o /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_baseline_beir.%J.txt
#SBATCH -e /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_baseline_beir.%J.txt
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=4         
#SBATCH --time=20:20:00              
#SBATCH --account=IscrC_MOSAICIR      
#SBATCH --partition=boost_usr_prod  #lrd_all_serial boost_usr_prod
#SBATCH --qos=normal #normal boost_qos_dbg
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL

export HF_HOME=/leonardo_work/IscrC_MOSAICIR/Braga/hf_home
export HF_DATASETS_CACHE=/leonardo_work/IscrC_MOSAICIR/Braga/hf_datasets_cache
export TRANSFORMERS_CACHE=/leonardo_work/IscrC_MOSAICIR/Braga/hf_models_cache
export TMPDIR=/leonardo_work/IscrC_MOSAICIR/Braga/tmp
export IR_DATASETS_HOME=/leonardo_scratch/fast/IscrC_MOSAICIR/ir_datasets

export PYTERRIER_HOME=/leonardo_work/IscrC_MOSAICIR/Braga/pyterrier_home
export M2_HOME=/leonardo_work/IscrC_MOSAICIR/Braga/m2
export MAVEN_OPTS="-Dmaven.repo.local=/leonardo_work/IscrC_MOSAICIR/Braga/m2"
export JAVA_TOOL_OPTIONS="-Djava.io.tmpdir=/leonardo_work/IscrC_MOSAICIR/Braga/tmp"
mkdir -p /leonardo_work/IscrC_MOSAICIR/Braga/pyterrier_home
mkdir -p /leonardo_work/IscrC_MOSAICIR/Braga/m2

mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE $TMPDIR $IR_DATASETS_HOME

module load profile/deeplrn
module load cineca-ai
source /leonardo_work/IscrC_MOSAICIR/Braga/tv_env/bin/activate

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "quora/test" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "quora/test" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "quora/test" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "quora/test" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "quora/test" --model_name "retromae_distil"


python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "hotpotqa/test" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "hotpotqa/test" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "hotpotqa/test" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "hotpotqa/test" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "hotpotqa/test" --model_name "retromae_distil"


#    done
#done