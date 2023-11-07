## DL params
export MINIBS=128
export TENSOR_MODEL_PARALLEL=4   #  training.model.tensor_model_parallel_size
export PIPELINE_MODEL_PARALLEL=8 #  training.model.pipeline_model_parallel_size
export DGXNNODES="${DGXNNODES:-96}"
#=======================================================================
## System run parms
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_MINUTES=90
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1635
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1470
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 3) # 33% longer walltime
fi
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_fp8.sh
export MICRO_BATCH_SIZE=1

export TP_COMM_OVERLAP=True
