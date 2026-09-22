#!/usr/bin/env bash
#SBATCH --job-name=524B-rand_perturb_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --time=0-08:00:00
#SBATCH --output=rand_perturb_exp/%x-%j.out
#SBATCH --error=rand_perturb_exp/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p logs-eval-slurm
echo "======= Conda and CUDA ======="

module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate duo

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "================================"


model_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt"

dataset_name="owt"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

done


dataset_name="wikitext103"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/wikitext103_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

done



dataset_name="pubmed"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/pubmed_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

    
done


dataset_name="lm1b"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/lm1b_test.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

done




dataset_name="arxiv"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/arxiv_valid.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

  
done





dataset_name="ag_news"
dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/ag_news_test.txt"
for p in 0.1 0.2 0.3 0.4
do
    python -u random_perturbation_experiment.py \
        mode=sample_eval \
        data=openwebtext-split \
        data.cache_dir="/home/nafez/scratch/remdm-shortcut-removal/data" \
        model=small \
        algo=mdlm \
        noise=log-linear \
        sampling.predictor=psi \
        sampling.steps=256 \
        sampling.p_nucleus=0.9 \
        sampling.num_sample_batches=1 \
        eval.checkpoint_path=$model_path \
        loader.eval_batch_size=8 \
        sampling.psi.time_profile=linear \
        sampling.psi.high_mode=max-rescale-0.05 \
        sampling.psi.middle_mode=max-rescale-0.05 \
        sampling.psi.low_mode=max-rescale-0.05 \
        sampling.psi.high_frac=0.0 \
        sampling.psi.middle_frac=0.0 \
        +shortcut_removal=True \
        +latent_noise=False \
        +use_trained_scaling_factor=False \
        +activate_nvib_noise=False \
        eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
        +dataset_path=$dataset_path \
        +chunk_size=512 \
        +corruption_prob=$p \
        +batch_size=64 \
        +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/coef-0-1-${dataset_name}_p-${p}.json"

done
