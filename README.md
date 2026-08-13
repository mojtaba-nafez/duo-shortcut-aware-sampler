# Random Word Perturbation Experiments

This section reproduces the random word perturbation experiments reported in the paper. The script evaluates the robustness of different diffusion language models under token-level perturbations.

## MDLM

```bash
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
    eval.checkpoint_path="/home/nafez/scratch/remdm-shortcut-removal/weights/mdlm.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=True
```

## SEDD


```bash
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
    eval.checkpoint_path="/home/nafez/scratch/duo/weights/sedd.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=True
```

## DOU


```bash
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
    eval.checkpoint_path="/home/nafez/scratch/duo/weights/duo.ckpt" \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0
```


# Running Experiments on the EPFL RCP Cluster

The following example launches a DUO evaluation job using the PSI sampler configuration:


```bash
runai submit \
  --name duo-rescale \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.3 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate remdm && \
    cd /scratch/mnafez/duo && \
    bash slurm_scripts/psi_samplers/owt/duo_max_rescale_eta.sh
    "
```


# Eval (Activate Random-Noise or Diagonal-Masking in Middle Layers)

## base sampler
```bash
python main.py \
  mode=sample_eval \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/weights/duo.ckpt \
  sampling.steps=1024 \
  sampling.num_sample_batches=1 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False
```


## Ψ-SAMPLER

```bash
python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=/idiap/temp/mnafez/research/duo/data \
    model=small \
    algo=duo_base \
    noise=log-linear \
    sampling.predictor=psi \
    sampling.steps=1024 \
    sampling.p_nucleus=0.9 \
    sampling.num_sample_batches=2 \
    eval.checkpoint_path=/idiap/temp/mnafez/research/duo/weights/duo.ckpt \
    loader.eval_batch_size=8 \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=False \
    +latent_noise=True
```


# Training

## 4 GPU
```bash
 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2 algo.curriculum.start=0   algo.curriculum.end=34000 +shortcut_removal=False  +latent_noise=False trainer.max_steps=68000 checkpointing.resume_from_ckpt=false model.nvib_layers=[4,6,8] trainer.val_check_interval=12000 trainer.limit_val_batches=1000 trainer.devices=4  loader.num_workers=16
```
+ trainer.max_steps=68000    --> number of optimizer step --> (num of GPUs)*(Per Batch Size)*(acuumulation)
+ trainer.val_check_interval --> based on number of dataloader batch --> (num of GPUs)*(Per Batch Size)
+ Epoch 0:   0%|▏         | 80/68360 [00:26<6:22:36,  2.97it/s, v_num=wt_1] --> 68360: total dataloader batch (num of GPUs)*(Per Batch Size)

17.8B :    68000*512*512 = ~17.8B

## 8 GPU
```bash
 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2 algo.curriculum.start=0   algo.curriculum.end=34000 +shortcut_removal=False  +latent_noise=False trainer.max_steps=68000 checkpointing.resume_from_ckpt=false model.nvib_layers=[4,6,8] trainer.val_check_interval=6000 trainer.limit_val_batches=1000 trainer.devices=4 trainer.num_nodes=2  loader.num_workers=16
```






sbatch --environment=gidd --nodes=2 -A a0236 --time=0-00:20:00 slurm_scripts/cscs/training/duo_owt_8gpu.sh




<!-- 
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2 algo.curriculum.start=0   algo.curriculum.end=50 +shortcut_removal=False  +latent_noise=False trainer.max_steps=2 checkpointing.resume_from_ckpt=false model.nvib_layers=[] trainer.val_check_interval=15 trainer.limit_val_batches=0 loader.num_workers=16
-->

CUDA_VISIBLE_DEVICES=0,1 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt-debug   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2   algo.curriculum.start=0   algo.curriculum.end=50   +shortcut_removal=False   +latent_noise=False   trainer.max_steps=2   checkpointing.resume_from_ckpt=false   model.nvib_layers=[]   trainer.limit_val_batches=0   loader.num_workers=1   trainer.devices=2   trainer.accumulate_grad_batches=4   trainer.log_every_n_steps=1



 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt-debug   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2   algo.curriculum.start=0   algo.curriculum.end=50   +shortcut_removal=False   +latent_noise=False   trainer.max_steps=2   checkpointing.resume_from_ckpt=false   model.nvib_layers=[]   trainer.limit_val_batches=0   loader.num_workers=1   trainer.devices=4    trainer.log_every_n_steps=1



 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main \
  loader.batch_size=64 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  wandb.name=duo-owt-multigpu-test \
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
  algo.curriculum.end=50 \
  +shortcut_removal=False \
  +latent_noise=False \
  trainer.max_steps=2 \
  checkpointing.resume_from_ckpt=false \
  model.nvib_layers=[4,6,8] \
  trainer.limit_val_batches=2 \
  loader.num_workers=8 \
  trainer.devices=4 \
  trainer.accumulate_grad_batches=2 \
  trainer.log_every_n_steps=1



 CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2 algo.curriculum.start=0   algo.curriculum.end=34000 +shortcut_removal=False  +latent_noise=False trainer.max_steps=81 checkpointing.resume_from_ckpt=false model.nvib_layers=[4,6,8] trainer.val_check_interval=80 trainer.limit_val_batches=1000 trainer.devices=4  loader.num_workers=16
