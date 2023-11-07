#!/bin/bash
set -x 
source config_XE8640x4H100.sh

export DATADIR=/mnt/data/openimages_ds/open-images-v6-mlperf
export LOGDIR=`pwd`/results_XE8640x4xH100
CONT=cd348636d7c0 ./run_with_docker.sh
