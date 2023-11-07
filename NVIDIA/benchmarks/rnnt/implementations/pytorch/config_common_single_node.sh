# Reasonable defaults for single node jobs
export DGXNNODES=1
export DATA_CPU_THREADS=16

: ${SINGLE_EXP_WALLTIME:=45}
export WALLTIME=${WALLTIME:-$(( ${NEXP:-1} * ${SINGLE_EXP_WALLTIME} ))}

## Opt flags
export MULTILAYER_LSTM=true
export MIN_SEQ_SPLIT_LEN=20
export PRE_SORT_FOR_SEQ_SPLIT=true

