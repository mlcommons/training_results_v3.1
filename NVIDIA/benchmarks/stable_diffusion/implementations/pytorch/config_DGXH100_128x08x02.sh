export DGXNNODES=128
export DGXNGPU=8
export BATCHSIZE=2
export CONFIG_MAX_STEPS=2250
export CHECKPOINT_STEPS=250
export SBATCH_NETWORK=sharp
# %%

## Set clocks and walltime for maxQ and minEDP runs
if [[ "${SET_MAXQ_CLK:-0}" == "1" ]]; then
  export MAXQ_CLK=930
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 2) # 50% longer walltime
elif [[ "${SET_MINEDP_CLK:-0}" == "1" ]]; then
  export MINEDP_CLK=1230
  WALLTIME_MINUTES=$(expr ${WALLTIME_MINUTES} + ${WALLTIME_MINUTES} / 3) # 33% longer walltime
fi

export CHECK_COMPLIANCE=0

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# CALCULATE LR as 1.25e-7 * DGXNNODES * DGXNGPU * BATCHSIZE
BASE_LR="0.0000001"
export LEARNING_RATE=$(echo "$BASE_LR * $DGXNNODES * $DGXNGPU * $BATCHSIZE" | bc -l)

: "${WALLTIME:=135}"
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME} + 5 ))
