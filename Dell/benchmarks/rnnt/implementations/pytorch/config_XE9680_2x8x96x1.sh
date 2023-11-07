# Single-node Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_XE9680_multinode.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh

export DGXNNODES=2
export BATCHSIZE=96
export EVAL_BATCHSIZE=169

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh

export WALLTIME=UNLIMITED
