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
    sampling.psi.middle_frac=0.0
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
    sampling.psi.middle_frac=0.0
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

