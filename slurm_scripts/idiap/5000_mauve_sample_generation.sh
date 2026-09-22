#!/usr/bin/env bash
#SBATCH --job-name=greedy-tail-ablation-mauve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --time=0-15:00:00
#SBATCH --output=logs-eval-slurm/%x-%j.out
#SBATCH --error=logs-eval-slurm/%x-%j.err
#SBATCH --requeue

set -e

mkdir -p logs-eval-slurm
mkdir -p greedy-tail-ablation-mauve-result
echo "======= Conda and CUDA ======="

module load CUDA

source /idiap/temp/mnafez/miniconda3/etc/profile.d/conda.sh
conda activate duo

echo "Python: $(which python)"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC: $(which nvcc)"
echo "================================"

# for greedy_step in 1 3 6; do


B=512
for greedy_step in 0 9; do
    b_update=$((B - greedy_step))

    python main.py \
    mode=sample_eval \
    data=openwebtext-split \
    algo=duo_base \
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt" \
    sampling.steps=$b_update \
    +wandb.offline=true \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    model.length=512 \
    loader.eval_batch_size=8 \
    loader.batch_size=8 \
    sampling.num_sample_batches=625 \
    eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
    sampling.noise_removal=greedy \
    +sampling.noise_removal_steps=$greedy_step \
    eval.generated_samples_path="$PWD/greedy-tail-ablation-mauve-result/duo-5000-clean-loss-term-greedy-${greedy_step}.json"
done

# for greedy_step in 1 3 6; do


B=512
for greedy_step in 0 9; do
    b_update=$((B - greedy_step))

    python main.py \
    mode=sample_eval \
    data=openwebtext-split \
    algo=duo_base \
    eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt" \
    sampling.steps=$b_update \
    +wandb.offline=true \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=False \
    +activate_nvib_noise=False \
    model.length=512 \
    loader.eval_batch_size=8 \
    loader.batch_size=8 \
    sampling.num_sample_batches=625 \
    eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
    sampling.noise_removal=greedy \
    +sampling.noise_removal_steps=$greedy_step \
    eval.generated_samples_path="$PWD/greedy-tail-ablation-mauve-result/duo-5000-original-greedy-${greedy_step}.json"
done