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
    +shortcut_removal=True \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
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
    +shortcut_removal=True \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
```

## DOU
/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term/last.ckpt

/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/duo-17B/last.ckpt

/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent/last.ckpt

/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term-time-dependent-0-02/last.ckpt

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
    +dataset_path="/idiap/temp/mnafez/research/Score-Entropy-Discrete-Diffusion/owt_valid.txt"
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
  +latent_noise=False \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
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
    +latent_noise=True \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
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


```bash
sbatch --environment=gidd --nodes=2 -A a0236 --time=0-00:20:00 slurm_scripts/cscs/training/duo_owt_8gpu.sh
```

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m main   loader.batch_size=64   loader.eval_batch_size=32   data=openwebtext-split   wandb.name=duo-owt   model=small   algo=duo   model.length=512   algo.curriculum.mode=poly9   algo.curriculum.gumbel_tau_log10_start=-3.0   algo.curriculum.gumbel_tau_log10_end=-3.0   algo.curriculum.gamma_min=-3.55   algo.curriculum.gamma_max=-1.85   algo.curriculum.top_k=2 algo.curriculum.start=0   algo.curriculum.end=34000 +shortcut_removal=False  +latent_noise=False trainer.max_steps=68000 checkpointing.resume_from_ckpt=false model.nvib_layers=[4,6,8] trainer.val_check_interval=12000 trainer.limit_val_batches=1000 trainer.devices=1  loader.num_workers=16 +use_trained_scaling_factor=False +activate_nvib_noise=False
```


# Evaluation

## Duo: Base Sampler

```bash
python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path="/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/duo-17B-clean-loss-term/last.ckpt" \
  sampling.steps=512 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=False \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=16 \
  sampling.num_sample_batches=16 \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
```

## Duo: PSI Sampler

```bash
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
    sampling.num_sample_batches=16 \
    loader.eval_batch_size=8 \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
```


## Duo + NVIB: Base Sampler

```bash
python main.py \
  mode=sample_eval \
  data=openwebtext-split \
  algo=duo_base \
  eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/gidd-nvib-17B/last.ckpt \
  sampling.steps=512 \
  sampling.noise_removal=greedy \
  +wandb.offline=true \
  model.nvib_layers=[4,6,8] \
  +shortcut_removal=False \
  +latent_noise=False \
  +use_trained_scaling_factor=True \
  +activate_nvib_noise=False \
  model.length=512 \
  loader.eval_batch_size=8 \
  loader.batch_size=16 \
  sampling.num_sample_batches=16 \
  eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
```


## Duo + NVIB: PSI Sampler

```bash
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
    eval.checkpoint_path=/idiap/temp/mnafez/research/duo/our-cscs-trained-checkpoints/our-pretrain-weights/gidd-nvib-17B/last.ckpt \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-0.05 \
    sampling.psi.middle_mode=max-rescale-0.05 \
    sampling.psi.low_mode=max-rescale-0.05 \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    model.nvib_layers=[4,6,8] \
    +shortcut_removal=False \
    +latent_noise=False \
    +use_trained_scaling_factor=True \
    +activate_nvib_noise=False \
    model.length=512 \
    sampling.num_sample_batches=16 \
    loader.eval_batch_size=8 \
    eval.gen_ppl_eval_model_name_or_path="google/gemma-2-9b"
```
