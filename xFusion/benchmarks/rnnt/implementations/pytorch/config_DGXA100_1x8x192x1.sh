# Single-node config (since v2.0)

source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh
source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh

export BATCHSIZE=192
export EVAL_BATCHSIZE=338

export AUDIO_RESAMPLING_DEVICE=gpu

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
