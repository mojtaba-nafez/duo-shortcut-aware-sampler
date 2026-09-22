#!/usr/bin/env bash
#SBATCH --job-name=duo_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --mem=80G
#SBATCH --time=0-14:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
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



python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=512 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"



python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term/last.ckpt \
  sampling.steps=512 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"



python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
  sampling.steps=512 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"

#####################################
#####################################
#####################################


python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=/idiap/temp/mnafez/research/duo/data \
    model=small \
    algo=duo_base \
    noise=log-linear \
    sampling.predictor=psi \
    sampling.steps=512 \
    sampling.p_nucleus=0.9 \
    eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
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
    model.length=512 \
    loader.eval_batch_size=8 \
    sampling.num_sample_batches=128 \
    loader.batch_size=8 \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"



python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=/idiap/temp/mnafez/research/duo/data \
    model=small \
    algo=duo_base \
    noise=log-linear \
    sampling.predictor=psi \
    sampling.steps=512 \
    sampling.p_nucleus=0.9 \
    eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term/last.ckpt \
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
    model.length=512 \
    loader.eval_batch_size=8 \
    sampling.num_sample_batches=128 \
    loader.batch_size=8 \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"


python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=/idiap/temp/mnafez/research/duo/data \
    model=small \
    algo=duo_base \
    noise=log-linear \
    sampling.predictor=psi \
    sampling.steps=512 \
    sampling.p_nucleus=0.9 \
    eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
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
    model.length=512 \
    loader.eval_batch_size=8 \
    sampling.num_sample_batches=128 \
    loader.batch_size=8 \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"