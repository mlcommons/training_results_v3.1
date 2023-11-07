## System config params
export DGXNGPU=4
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}

## Data Paths
export DATADIR="/mnt/data/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
export DATADIR_PHASE2="/mnt/data/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
export EVALDIR="/mnt/data/bert3.0/hdf5/eval_varlength/"
export CHECKPOINTDIR="./ci_checkpoints"
export RESULTSDIR="./results"
#using existing checkpoint_phase1 dir
export CHECKPOINTDIR_PHASE1="/mnt/data/bert3.0/phase1"
export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"


DATADIR_PHASE2_PACKED="/mnt/data/bert3.0/packed_data/"
export NCCL_SOCKET_IFNAME=
#
#
#export DATADIR="/mnt/data/training3.1/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
#export DATADIR_PHASE2="/mnt/data/training3.1/bert3.0/hdf5/training-4320/hdf5_4320_shards_varlength"
#export EVALDIR="/mnt/data/training3.1/bert3.0/hdf5/eval_varlength/"
#export CHECKPOINTDIR="./ci_checkpoints"
#export RESULTSDIR="./results"
##using existing checkpoint_phase1 dir
#export CHECKPOINTDIR_PHASE1="/mnt/data/training3.1/bert3.0/phase1"
##export UNITTESTDIR="/lustre/fsw/mlperf/mlperft-bert/unit_test"
#export DATADIR_PHASE2_PACKED="/mnt/data/training3.1/bert3.0/packed_data/"
#export NCCL_SOCKET_IFNAME=



























