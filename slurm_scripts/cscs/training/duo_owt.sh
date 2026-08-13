#!/usr/bin/env bash

#SBATCH --job-name=duo-owt
#SBATCH --account=a0236
#SBATCH --partition=normal

# 2 nodes × 4 GH200 = 8 GPUs total
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

# 16 DataLoader workers per rank -> give each rank CPU headroom
#SBATCH --cpus-per-task=32

# Clariden normal partition maximum is 12 hours
#SBATCH --time=0-12:00:00

#SBATCH --output=logs-train-slurm/%x-%j.out
#SBATCH --error=logs-train-slurm/%x-%j.err

set -euo pipefail

# ============================================================
# Project / environment
# ============================================================

PROJECT_DIR="/mnt/home/NLU/duo"
CONDA_SH="/mnt/home/miniconda3/etc/profile.d/conda.sh"

cd "${PROJECT_DIR}"

# ============================================================
# Distributed / debugging environment
# ============================================================

export PYTHONUNBUFFERED=1

# Keep this WARN for production.
# Change to INFO if debugging NCCL.
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Lightning can derive these from Slurm, but setting them
# explicitly makes the setup obvious and deterministic.
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)

export MASTER_PORT
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "============================================================"
echo "DUO distributed training"
echo "Job ID:             ${SLURM_JOB_ID}"
echo "Nodes:              ${SLURM_JOB_NUM_NODES}"
echo "Node list:          ${SLURM_JOB_NODELIST}"
echo "Tasks per node:     ${SLURM_NTASKS_PER_NODE}"
echo "Total tasks:        ${SLURM_NTASKS}"
echo "MASTER_ADDR:        ${MASTER_ADDR}"
echo "MASTER_PORT:        ${MASTER_PORT}"
echo "============================================================"

scontrol show hostnames "${SLURM_JOB_NODELIST}"

# ============================================================
# Launch
# ============================================================
#
# IMPORTANT:
#   Slurm launches 4 ranks per node.
#   Lightning:
#       trainer.devices=4
#       trainer.num_nodes=2
#
# Total DDP world size = 4 × 2 = 8 GPUs
#
# Global batch:
#       64 × 8 GPUs × accumulation 1 = 512
#
# ============================================================

srun -ul --cpu-bind=cores bash -c '

    source /mnt/home/miniconda3/etc/profile.d/conda.sh
    conda activate duo

    cd /mnt/home/NLU/duo

    echo "------------------------------------------------------------"
    echo "Host:                 $(hostname)"
    echo "SLURM_PROCID:         ${SLURM_PROCID}"
    echo "SLURM_LOCALID:        ${SLURM_LOCALID}"
    echo "SLURM_NODEID:         ${SLURM_NODEID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "Python:               $(which python)"
    echo "------------------------------------------------------------"

    python -u -m main \
        loader.batch_size=64 \
        loader.eval_batch_size=32 \
        data=openwebtext-split \
        wandb.name=duo-owt \
        model=small \
        algo=duo \
        model.length=512 \
        algo.curriculum.mode=poly9 \
        algo.curriculum.gumbel_tau_log10_start=-3.0 \
        algo.curriculum.gumbel_tau_log10_end=-3.0 \
        algo.curriculum.gamma_min=-3.55 \
        algo.curriculum.gamma_max=-1.85 \
        algo.curriculum.top_k=2 \
        algo.curriculum.start=0 \
        algo.curriculum.end=34000 \
        +shortcut_removal=False \
        +latent_noise=False \
        checkpointing.resume_from_ckpt=false \
        trainer.devices=4 \
        trainer.num_nodes=2 \
        trainer.accumulate_grad_batches=1 \
        loader.num_workers=16 \
        trainer.val_check_interval=6000 \
        trainer.limit_val_batches=1000 \
        trainer.max_steps=68000
'