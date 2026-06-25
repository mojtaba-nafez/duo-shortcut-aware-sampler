# DUO psi-sampler with max-rescale-eta mode (ReMDM rescale)

NUM_STEPS=1024
ETA=0.05
NUCLEUS_P=0.9
NOISE=log-linear
CHECKPOINT_PATH="/scratch/mnafez/duo/weights/duo.ckpt"
DATA_CACHE_DIR="/scratch/mnafez/remdm-shortcut-removal/data"

EVAL_BATCH_SIZE=8
NUM_SAMPLE_BATCHES=30

echo "Current directory:"
pwd

echo "Files:"
ls -lah



python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=$DATA_CACHE_DIR \
    model=small \
    algo=duo_base \
    noise=$NOISE \
    sampling.predictor=psi \
    sampling.steps=$NUM_STEPS \
    sampling.p_nucleus=$NUCLEUS_P \
    sampling.num_sample_batches=$NUM_SAMPLE_BATCHES \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    loader.eval_batch_size=$EVAL_BATCH_SIZE \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-$ETA \
    sampling.psi.middle_mode=max-rescale-$ETA \
    sampling.psi.low_mode=max-rescale-$ETA \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=false



python -u -m main \
    mode=sample_eval \
    data=openwebtext-split \
    data.cache_dir=$DATA_CACHE_DIR \
    model=small \
    algo=duo_base \
    noise=$NOISE \
    sampling.predictor=psi \
    sampling.steps=$NUM_STEPS \
    sampling.p_nucleus=$NUCLEUS_P \
    sampling.num_sample_batches=$NUM_SAMPLE_BATCHES \
    eval.checkpoint_path=$CHECKPOINT_PATH \
    loader.eval_batch_size=$EVAL_BATCH_SIZE \
    sampling.psi.time_profile=linear \
    sampling.psi.high_mode=max-rescale-$ETA \
    sampling.psi.middle_mode=max-rescale-$ETA \
    sampling.psi.low_mode=max-rescale-$ETA \
    sampling.psi.high_frac=0.0 \
    sampling.psi.middle_frac=0.0 \
    +shortcut_removal=true