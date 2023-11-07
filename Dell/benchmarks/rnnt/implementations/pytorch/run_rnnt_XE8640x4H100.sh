#!/bin/bash

set -x 

export DATADIR="/mnt/data/rnnt_new"
export METADATA_DIR="/mnt/data/rnnt_new/metadatadir/"
export SENTENCEPIECES_DIR="/mnt/data/rnnt_new/sentencepieces/"

source config_XE8640x4H100.sh

NEXP=10
DGXSYSTEM=XE8640x4H100
CONT=9f0b322c9478 ./run_with_docker.sh
