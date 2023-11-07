# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark


```
$: cd /home/mtroaca/training/bert/scripts
$: source config_DGXH100_1x4x48x1_pack.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/bert:20230926.pytorch DATADIR_PHASE2=/home/mtroaca/training/bert/data/packed_data EVALDIR=/home/mtroaca/training/bert/data/hdf5/eval_varlength CHECKPOINTDIR=/home/mtroaca/training/bert/checkpoints CHECKPOINTDIR_PHASE1=/home/mtroaca/training/bert/data/phase1 LOGDIR=/home/mtroaca/training/bert/logs bash run_with_docker.sh
```

