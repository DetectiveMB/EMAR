#!/bin/bash
#SBATCH -J /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_cqa
#SBATCH -o /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_cqa.%J.txt
#SBATCH -e /leonardo_work/IscrC_MOSAICIR/Braga/Braga_results/run_prova_smart_cqa.%J.txt
#SBATCH --nodes=1                    
#SBATCH --ntasks-per-node=4         
#SBATCH --time=5:20:00 
#SBATCH --gres=gpu:1             
#SBATCH --account=IscrC_MOSAICIR      
#SBATCH --partition=boost_usr_prod 
#SBATCH --qos=normal #normal boost_qos_dbg
#SBATCH --mem=30GB
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


python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/android" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/android" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/android" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/android" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/android" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/gis" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/gis" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/gis" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/gis" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/gis" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/mathematica" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/mathematica" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/mathematica" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/mathematica" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/mathematica" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/programmers" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/programmers" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/programmers" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/programmers" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/programmers" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/stats" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/stats" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/stats" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/stats" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/stats" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/unix" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/unix" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/unix" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/unix" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/unix" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/webmasters" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/webmasters" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/webmasters" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/webmasters" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/webmasters" --model_name "retromae_distil"

python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/wordpress" --model_name "BERT_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/wordpress" --model_name "Distilbert_msmarco"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/wordpress" --model_name "mpnet"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/wordpress" --model_name "mpnet_QA"
python /leonardo_work/IscrC_MOSAICIR/Braga/eval_beir.py --dataset "cqadupstack/wordpress" --model_name "retromae_distil"

