#!/usr/bin/env bash
#SBATCH --job-name=greedy-3-duo-clean-term-step-ablation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBATCH --time=0-09:00:00
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



echo "=============== Step: 16 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=13 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=13 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3

echo "=============== Step: 32 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=29 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=29 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3



echo "=============== Step: 64 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=61 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=61 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


echo "=============== Step: 128 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=125 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=125 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


echo "=============== Step: 256 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=253 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=253 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3



echo "=============== Step: 512 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=509 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=509 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


echo "=============== Step: 1024 ==============="

python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt \
  sampling.steps=1021 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3


python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt \
  sampling.steps=1021 \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=8 \
  sampling.num_sample_batches=128 \
  eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
  sampling.noise_removal=greedy \
  +sampling.noise_removal_steps=3



# #!/usr/bin/env bash


# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=11 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5

# echo "=============== Step: 32 ==============="


# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=27 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5



# echo "=============== Step: 64 ==============="


# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=59 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5


# echo "=============== Step: 128 ==============="

# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=123 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5


# echo "=============== Step: 256 ==============="


# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=251 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5



# echo "=============== Step: 512 ==============="

# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=507 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5


# echo "=============== Step: 1024 ==============="

# python main.py \
#   mode=sample_eval \
#   data=openwebtext-split \
#   algo=duo_base \
#   eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt \
#   sampling.steps=1019 \
#   +wandb.offline=true \
#   +shortcut_removal=False \
#   +latent_noise=False \
#   +use_trained_scaling_factor=False \
#   +activate_nvib_noise=False \
#   model.length=512 \
#   loader.eval_batch_size=8 \
#   loader.batch_size=8 \
#   sampling.num_sample_batches=128 \
#   eval.gen_ppl_eval_model_name_or_path="gpt2-large" \
#   sampling.noise_removal=greedy \
#   +sampling.noise_removal_steps=5

