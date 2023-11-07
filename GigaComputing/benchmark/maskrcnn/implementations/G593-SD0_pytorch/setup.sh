#!/bin/bash 
cd ../pytorch
docker build --pull -t mlperf_trainingv3.1-gigacomputing:maskrcnn .
