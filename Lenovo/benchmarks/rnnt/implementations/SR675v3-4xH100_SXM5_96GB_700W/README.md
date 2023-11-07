# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark


```
$: cd /home/mtroaca/training/rnnt/scripts
$: source config_DGXH100_1x4x192x1.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/rnnt:20230926.pytorch DATADIR=/home/mtroaca/training/rnnt/data METADATA_DIR=/home/mtroaca/training/rnnt/metadata SENTENCEPIECES_DIR=/home/mtroaca/training/rnnt/sentencepieces LOGDIR=/home/mtroaca/training/rnnt/logs NEXP=10 bash run_with_docker.sh
```

