
# Main

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

