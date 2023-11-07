# v3.0 8n Hopper config

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_multi_node.sh

export DGXNNODES=8
export BATCHSIZE=32
export EVAL_BATCHSIZE=43

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
