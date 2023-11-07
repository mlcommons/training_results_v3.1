# v3.0 64 nodes GH submission

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=7}
source $(dirname ${BASH_SOURCE[0]})/config_common_multi_node.sh

export DGXNNODES=64
export BATCHSIZE=8
export EVAL_BATCHSIZE=6  # ceil(2703 / $DGXNGPU / $DGXNNODES)

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true
export NUM_CG=50

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh

