#!/bin/bash

set -x 


source config_XE8640x4H100_SXM_80GB.sh

export NEXP=10
#export NEXP=12
export CUDA_VISIBLE_DEVICES=0,1,2,3

export LOGDIR=/home/rakshith/mlperf_training_3.1/bert/scripts/results_XE8640x4H100
DGXSYSTEM=XE8640x4H100_SXM_80GB CONT=d44c01427c87   ./run_with_docker.sh


