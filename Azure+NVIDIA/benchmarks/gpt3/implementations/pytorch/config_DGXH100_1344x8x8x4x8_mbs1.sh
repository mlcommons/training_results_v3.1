export MINIBS=8
export TENSOR_MODEL_PARALLEL=4
export PIPELINE_MODEL_PARALLEL=8
export DGXNNODES=1344
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME_MINUTES=190
export WALLTIME=$(( (${NEXP:-1} * WALLTIME_MINUTES) ))
source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_fp8.sh
export MICRO_BATCH_SIZE=1
export TP_COMM_OVERLAP=True
