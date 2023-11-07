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

export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
export OMPI_MCA_btl=^openib
export CUDNN_FORCE_KERNEL_INIT=1

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

WALLTIME_MINUTES=20
if [ ${MLPERF_POWER_TRAIN_AFTER_RUN_STOP} -gt 0 ]; then
  WALLTIME_MINUTES=$((${WALLTIME_MINUTES} + 10))
  export EXTRA_PARAMS="${EXTRA_PARAMS} -stt 10 "
  export MAX_EPOCHS=100000
fi
if [[ "${SET_MAXQ_CLK}" == "1" ]] || [[ "${SET_MINEDP_CLK}" == "1" ]]; then
  export MAXQ_CLK=1125
  export MINEDP_CLK=1485
fi
export WALLTIME=$(( 10 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=52
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1


export NCCL_SOCKET_IFNAME=
