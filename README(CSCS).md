# CSCS-Base Document

# Training:

## Intractive Session Training:

```
srun --environment=gidd  -A a0236  --pty bash
source ~/miniconda3/bin/activate
conda activate duo
```

```
srun --jobid=3054265 --overlap --export=ALL --pty bash -i
```


# Installation:

```bash
conda create -n duo python=3.11 -y
conda activate duo

conda install nvidia/label/cuda-12.4.0::cuda-toolkit -y
pip install ninja psutil
MAX_JOBS=4 pip install flash_attn==2.7.4.post1 --no-build-isolation
```

```bash
mkdir -p $SCRATCH/.cache
ln -s $SCRATCH/.cache .

mkdir -p $SCRATCH/outputs
ln -s $SCRATCH/outputs .
```
