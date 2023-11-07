# Single-node Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=25}
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh

export BATCHSIZE=192
export EVAL_BATCHSIZE=338

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
