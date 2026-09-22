#!/usr/bin/env bash
#SBATCH --job-name=rand_perturb_exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --time=0-06:00:00
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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.1 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/clean_loss_p-0-1.json"

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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.2 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/clean_loss_p-0-2.json"


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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.3 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/clean_loss_p-0-3.json"


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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.4 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/clean_loss_p-0-4.json"



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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.5 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/clean_loss_p-0-5.json"






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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/best.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.1 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/original_p-0-1.json"




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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/best.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.2 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/original_p-0-2.json"





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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/best.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.3 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/original_p-0-3.json"



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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/best.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.4 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/original_p-0-4.json"



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
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/best.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b" \
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt" \
    +chunk_size=512 \
    +corruption_prob=0.5 \
    +batch_size=64 \
    +save_path="/idiap/temp/mnafez/research/duo/rand_perturb_exp/original_p-0-5.json"


