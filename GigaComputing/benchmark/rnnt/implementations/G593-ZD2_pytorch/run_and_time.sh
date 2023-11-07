#!/bin/bash

cd ../pytorch
source config_G593-ZD2_1x8x192x1.sh
export CONT=mlperf_trainingv3.1-gigacomputing:rnnt
export LOGDIR=/path/to/folder
export DATADIR=/path/to/dataset
export METADATA_DIR=/path/to/tokenized 
export SENTENCEPIECES_DIR=/path/to/sentencepieces 

./run_with_docker.sh 
