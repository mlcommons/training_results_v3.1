# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1

The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark

```
cd /home/mtroaca/training/ssd/scripts
source config_DGXH100_001x04x032.sh
CONT=nvcr.io/nvdlfwea/mlperfv31/ssd:20230926.pytorch DATADIR=/dm7100f/test/data LOGDIR=/home/mtroaca/training/ssd/logs bash run_with_docker.shC: cd /scratch/training-v3.1/ssd/scripts
```
