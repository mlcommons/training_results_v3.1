export BATCHSIZE=64 #try 48
export CONFIG_MAX_STEPS=6000 #try 6250
#export CONFIG_MAX_STEPS=2000 #For faster troubleshooting on validation error
#export CHECKPOINT_STEPS=1333 #try 1250
export CHECKPOINT_STEPS=1000 #try 1250
export CHECK_COMPLIANCE=0

#export CONFIG_MAX_STEPS=200 #try 6250
#export CHECKPOINT_STEPS=100 #try 1250

export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )


#: "${WALLTIME:=235}"
export WALLTIME=$(( ${NEXP:-1} * ${WALLTIME} + 5 ))

## System config params
export DGXNNODES=1
export DGXNGPU=8
export DGXSOCKETCORES=56
export DGXNSOCKET=2
export DGXHT=2  # HT is on is 2, HT off is 1
export DATETIME=$(date +%y%m%d%H%M%S%N)

## DL params
# CALCULATE LR as 1.25e-7 * DGXNNODES * DGXNGPU * BATCHSIZE
BASE_LR="0.0000001"
export LEARNING_RATE=$(echo "$BASE_LR * $DGXNNODES * $DGXNGPU * $BATCHSIZE" | bc -l)

#export NCCL_DEBUG=INFO
#export NCCL_P2P_LEVEL=NVL
#export TORCH_DISTRIBUTED_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=


#For troubleshooting 
#export INFER_NUM_IMAGES=3752

