# Reasonable defaults for multi node jobs (>8 nodes)
export DATA_CPU_THREADS=8

: ${SINGLE_EXP_WALLTIME:=10}
export WALLTIME=${WALLTIME:-$(( 10 + ${NEXP:-1} * ${SINGLE_EXP_WALLTIME} ))}

## Opt flags
export MULTILAYER_LSTM=false

## network flags
export SBATCH_NETWORK=sharp
export NCCL_COLLNET_ENABLE=1
# export NCCL_DEBUG=INFO
