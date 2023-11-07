export DGXNNODES=1
export DGXNGPU=8
export BATCHSIZE=32
export CONFIG_MAX_STEPS=6000
export CHECKPOINT_STEPS=1000

# %%

export CHECK_COMPLIANCE=0

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

# CALCULATE LR as 1.25e-7 * DGXNNODES * DGXNGPU * BATCHSIZE
BASE_LR="0.0000001"
export LEARNING_RATE=$(echo "$BASE_LR * $DGXNNODES * $DGXNGPU * $BATCHSIZE" | bc -l)

: "${WALLTIME:=235}"
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME} + 5 ))
