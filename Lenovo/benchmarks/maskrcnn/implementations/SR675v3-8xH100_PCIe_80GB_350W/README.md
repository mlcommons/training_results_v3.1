# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark

```
$: cd /scratch/mtroaca/training/rcnn/scripts
$: source config_DGXH100.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/maskrcnn:20230926.pytorch DATADIR=/scratch/training-v3.1/rcnn/data LOGDIR=/scratch/training-v3.1/rcnn/logs PKLPATH=/scratch/training-v3.1/rcnn/data/coco2017/models bash run_with_docker.sh
```
