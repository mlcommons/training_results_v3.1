# Single-node Hopper config (since v2.1)

source $(dirname ${BASH_SOURCE[0]})/config_common_SYS-821GE-TNHR.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_benchmark.sh
: ${SINGLE_EXP_WALLTIME:=25}
EXTERNAL_WALLTIME=${WALLTIME:-}  # capture external WALLTIME before `config_common_single_node.sh` overrides it
source $(dirname ${BASH_SOURCE[0]})/config_common_single_node.sh

export BATCHSIZE=192
export EVAL_BATCHSIZE=338

export AUDIO_RESAMPLING_DEVICE=gpu
export DELAY_ENCODER=true

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_auto.sh

## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=1320
  SINGLE_EXP_WALLTIME=$(expr ${SINGLE_EXP_WALLTIME} + ${SINGLE_EXP_WALLTIME} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1695
  SINGLE_EXP_WALLTIME=$(expr ${SINGLE_EXP_WALLTIME} + ${SINGLE_EXP_WALLTIME} / 3) # 33% longer walltime
fi

## Power extension
: "${POWER_EXTENSION_MINUTES:=10}"
: "${POWER_EXTENSION_EPOCHS:=$EPOCH}"  # base run should already be longer than 10 minutes
# Extends run to number of minutes given by `POWER_EXTENSION_MINUTES` only if `MLPERF_POWER_TRAIN_AFTER_RUN_STOP` is 1
# EPOCH must be large enough to keep the benchmark running for $POWER_EXTENSION_MINUTES minutes
if [[ "${MLPERF_POWER_TRAIN_AFTER_RUN_STOP:-0}" == "1" ]]; then
  export EXTEND_RUN_TO_MINUTES=$POWER_EXTENSION_MINUTES
  export EPOCH=${POWER_EXTENSION_EPOCHS}
  # at least the number of minutes as POWER_EXTENSION_MINUTES + 5
  if [[ "${SINGLE_EXP_WALLTIME}" -lt $(( POWER_EXTENSION_MINUTES + 5 )) ]]; then
    SINGLE_EXP_WALLTIME=$(expr ${POWER_EXTENSION_MINUTES} + 5)
  fi
fi

export WALLTIME=${EXTERNAL_WALLTIME:-$(( ${NEXP:-1} * ${SINGLE_EXP_WALLTIME} ))}
