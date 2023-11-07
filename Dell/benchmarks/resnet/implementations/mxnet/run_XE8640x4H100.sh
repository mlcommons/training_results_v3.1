#!/bin/bash

set -x 

source config_XE8640x4H100.sh

export DATADIR=/mnt/data/training3.1/ilsvrc12_passthrough
export LOGDIR=/home/rakshith/mlperf_training_3.1/resnet/20230913/scripts/results_XE8640
CONT=ffc62891197e ./run_with_docker.sh

