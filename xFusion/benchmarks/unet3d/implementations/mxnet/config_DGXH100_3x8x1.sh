## DL params
export OPTIMIZER="nag"
export BATCH_SIZE="1"
export VAL_BATCH_SIZE="1"
export LR="2.0"
export LR_WARMUP_EPOCHS="1000"
export MAX_EPOCHS=${MAX_EPOCHS:-10000}
export START_EVAL_AT=1000
export EVALUATE_EVERY=20
export QUALITY_THRESHOLD="0.908"
export INPUT_BATCH_MULTIPLIER=4
export NUM_WORKERS=4
export EXTRA_PARAMS=${EXTRA_PARAMS:-"-sts -ucl "}
export PRECISION=${PRECISION:-"--static_cast -sls 8192 -gpf 32 --fp16in "}

# sharp disabled due to seg fault
# export SBATCH_NETWORK=sharp
export OMP_NUM_THREADS=1
export HOROVOD_CYCLE_TIME=0.1
export OMPI_MCA_btl=^openib

## System run parms
export DGXNNODES=3
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
WALLTIME_MINUTES=15
export WALLTIME=$(( 10 + (${NEXP:-1} * ${WALLTIME_MINUTES}) ))

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
