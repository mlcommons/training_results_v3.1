# 4n Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=30}
source $(dirname ${BASH_SOURCE[0]})/config_common_multi_node.sh

export DGXNNODES=4
export BATCHSIZE=64
export EVAL_BATCHSIZE=85

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
