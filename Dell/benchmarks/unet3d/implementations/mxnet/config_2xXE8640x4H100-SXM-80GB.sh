## DL params
export OPTIMIZER="nag"
export BATCHSIZE="7"
export VAL_BATCH_SIZE="4"
export LR="2.0"
export LR_WARMUP_EPOCHS="1000"
export MAX_EPOCHS=${MAX_EPOCHS:-10000}
export START_EVAL_AT=1000
export QUALITY_THRESHOLD="0.908"
export INPUT_BATCH_MULTIPLIER=4
export NUM_WORKERS=4
export EXTRA_PARAMS=${EXTRA_PARAMS:-"-sts -ucl "}
export PRECISION=${PRECISION:-"--static_cast -sls 32768 -gpf 4 --fp16in "}

#export SBATCH_NETWORK=sharp
export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
#export MXNET_HOROVOD_NUM_GROUPS=20
export OMPI_MCA_btl=^openib
#export NCCL_MAX_RINGS=4
#export NCCL_BUFFSIZE=2097152
#export NCCL_NET_GDR_READ=1
#export HOROVOD_FUSION_THRESHOLD=67108864
#export HOROVOD_NUM_NCCL_STREAMS=1
#export HOROVOD_BATCH_D2D_MEMCOPIES=1
#export HOROVOD_GROUPED_ALLREDUCES=1
#export MXNET_EXEC_ENABLE_ADDTO=0
export CUDNN_FORCE_KERNEL_INIT=1

## System run parms
export DGXNNODES=2
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME=UNLIMITED

## System config params
export DGXNGPU=4
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1  # HT is on is 2, HT off is 1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
