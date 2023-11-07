#!/bin/bash

cd ../pytorch
source config_G593-ZD2_1x8x48x1_pack.sh
export CONT=mlperf_trainingv3.1-gigacomputing:bert
export LOGDIR="</path/to/logdir>"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATADIR="<path/to/4320_shards_varlength/dir>"
export EVALDIR="<path/to/eval_varlength/dir>"
export DATADIR_PHASE2="<path/to/4320_shards_varlength/dir>"
export CHECKPOINTDIR="<path/to/result/checkpointdir>"
export CHECKPOINTDIR_PHASE1="<path/to/result/checkpointdir>"
export UNITTESTDIR="</path/to/pytorch/unit_test>"

./run_with_docker.sh
