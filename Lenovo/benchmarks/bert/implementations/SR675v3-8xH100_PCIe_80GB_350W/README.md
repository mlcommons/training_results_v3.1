# benchmark performed using the NVidia NGC containerized version of MLPerf training 3.1


The Lenovo MLPerf Training 3.1 submission on NVIDIA GPUs are made using the NVIDIA prepared
containers for MLPerf. For details on the source code we refer to the NVIDIA submission to
avoid duplicity.

Details of launch parameters and any other Lenovo particulars using this framework is outlined
in the README.md for each individual benchmark


```
$: cd /scratch/training-v3.1/bert/scripts
$: source config_DGXH100_1x8x48x1_pack.sh
$: CONT=nvcr.io/nvdlfwea/mlperfv31/bert:20230926.pytorch DATADIR_PHASE2=/scratch/training-v3.1/bert/data/packed_data EVALDIR=/scratch/training-v3.1/bert/data/hdf5/eval_varlength CHECKPOINTDIR=/scratch/training-v3.1/bert/checkpoints CHECKPOINTDIR_PHASE1=/scratch/training-v3.1/bert/data/phase1 LOGDIR=/scratch/training-v3.1/bert/logs NEXP=1 bash run_with_docker.sh

```

