#!/bin/bash
#nvidia-smi -lgc 1980 #GPU cores clocks
#nvidia-smi -ac 2619,1980 #GPU core and memory clocks
#cpupower frequency-set -g performance   

cd ../mxnet
source config_G593-ZD2_1x8x7.sh
export CONT=mlperf_trainingv3.1-gigacomputing:unet3d
export DATADIR=/path/to/dataset 
export LOGDIR=/path/to/results
./run_with_docker.sh

