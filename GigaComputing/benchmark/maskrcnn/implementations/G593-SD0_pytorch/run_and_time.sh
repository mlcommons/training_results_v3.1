#!/bin/bash

cd ../pytorch
source config_G593-ZD2.sh
export CONT=mlperf_trainingv3.1-gigacomputing:maskrcnn
export DATADIR=/path/to/preprocessed/data
export LOGDIR=/path/to/logfile
export PKLDIR=/path/to/folder
export coco_train=/path/to/folder
./run_with_docker.sh
