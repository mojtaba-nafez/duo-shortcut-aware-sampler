#!/bin/bash

export HYDRA_FULL_ERROR=1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --steps) steps="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --ckpt) ckpt="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

checkpoint_path=/share/kuleshov/ssahoo/flow-ode/distil-distil-vjrpZb-distillation-OWT/checkpoints
ckpt=0-50000

steps=${steps:-32}
seed=${seed:-1}

echo "Steps: $steps"
echo "Seed: $seed"
echo "ckpt: $ckpt"

python -u -m main \
  mode=sample_eval \
  seed=$seed \
  loader.batch_size=2 \
  loader.eval_batch_size=8 \
  data=openwebtext-split \
  algo=duo_base \
  model=small \
  eval.checkpoint_path=$checkpoint_path/$ckpt.ckpt \
  sampling.num_sample_batches=100 \
  sampling.steps=$steps \
  +wandb.offline=true \
  eval.generated_samples_path=$checkpoint_path/samples_ancestral/$seed-$steps-ckpt-$ckpt.json