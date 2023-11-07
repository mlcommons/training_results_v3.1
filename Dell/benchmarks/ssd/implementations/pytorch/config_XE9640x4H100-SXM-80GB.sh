#!/bin/bash

## DL params
export BATCHSIZE=${BATCHSIZE:-64}
export NUMEPOCHS=${NUMEPOCHS:-8}
export LR=${LR:-0.000085}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --dali-cpu-decode --master-weights'}

## System run params
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=60

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=56
export DGXNSOCKET=2
export DGXHT=1  # HT is on is 2, HT off is 1

## Replace memset kernels with explicit zero-out kernels
export CUDNN_FORCE_KERNEL_INIT=1
export NCCL_MAX_RINGS=4
export NCCL_SOCKET_IFNAME=^eno

