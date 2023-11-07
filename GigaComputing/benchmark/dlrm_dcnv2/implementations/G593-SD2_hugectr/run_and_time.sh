#!/bin/bash

cd ../hugectr
source config_G593-ZD2_1x8x6912.sh
export CONT=mlperf_trainingv3.1-gigacomputing:dlrmv2
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
./run_with_docker.sh
