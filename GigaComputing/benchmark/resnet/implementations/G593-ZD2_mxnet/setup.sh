#!/bin/bash 
cd ../mxnet
docker build --pull -t mlperf_trainingv3.1-gigacomputing:resnet .
