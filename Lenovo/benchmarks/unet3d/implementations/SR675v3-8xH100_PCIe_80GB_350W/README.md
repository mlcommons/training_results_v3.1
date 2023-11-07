# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark


```
$: cd /scratch/training-v3.1/3d-unet/scripts
$: source config_DGXH100_1x8x7.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/unet3d:20230926.mxnet DATADIR=/scratch/training-v3.1/3d-unet/data LOGDIR=/scratch/training-v3.1/3d-unet/logs NEXP=40 bash run_with_docker.sh
```

