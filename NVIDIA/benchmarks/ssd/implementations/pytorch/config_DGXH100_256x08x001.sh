#!/bin/bash
export SBATCH_NETWORK=sharp

## DL params
export BATCHSIZE=${BATCHSIZE:-1}
export NUMEPOCHS=${NUMEPOCHS:-12}
export LR=${LR:-0.000135}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --master-weights --dali-decoder-hw-load=0.99 --dali-input-batch-multiplier=16 --workers=32 --coco-threads=28 --eval-batch-size=8'}

## System run params
export DGXNNODES=256
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=20
if [ "${HANG_MONITOR_TIMEOUT-0}" -gt 0 ]; then
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${HANG_MONITOR_TIMEOUT}) # Extend run to debug hang
fi
export WALLTIME=$((${NEXP:-1} * ${WALLTIME_MINUTES}))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1

## Replace memset kernels with explicit zero-out kernels
export CUDNN_FORCE_KERNEL_INIT=1
