# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark


```
$: cd /scratch/training-v3.1/resnet50/scripts
$: source config_DGXH100.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/resnet:20230926.mxnet DATADIR=/scratch/training-v3.1/resnet50/data LOGDIR=/scratch/training-v3.1/resnet50/logs bash run_with_docker.sh
```

