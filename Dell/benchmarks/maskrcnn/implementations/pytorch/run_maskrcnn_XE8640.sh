#!/bin/bash
set -x 

source config_XE8640x4H100.sh

#export PKLPATH=/mnt/data1/training_ds/coco2017/pickled
#export DATADIR=/mnt/data1/training_ds
export PKLPATH=/mnt/data/coco2017/pickled
export DATADIR=/mnt/data

CONT=815746369624  LOGDIR=/home/rakshith/mlperf_training_3.1/maskrcnn/scripts/results_XE86404xH100 ./run_with_docker.sh

#CONT=4927e4a625bd  LOGDIR=/home/rakshith/mlperf_training_3.0/maskrcnn/scripts/results_R760xa4xH100 ./run_with_docker.sh # binding 
