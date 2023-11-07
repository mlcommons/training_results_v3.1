#!/bin/bash

cd ../mxnet
source config_G593-ZD2.sh
export CONT=mlperf_trainingv3.1-gigacomputing:resnet
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh ${bind_file}
