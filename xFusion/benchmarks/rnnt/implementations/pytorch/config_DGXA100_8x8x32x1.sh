# v2.0 efficient scale config (replacing 16x8x16x1)

## System config params
source $(dirname ${BASH_SOURCE[0]})/config_common_dgx.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_multi_node.sh
source $(dirname ${BASH_SOURCE[0]})/config_data_selene.sh

export DGXNNODES=8
export BATCHSIZE=32
export EVAL_BATCHSIZE=43

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh
