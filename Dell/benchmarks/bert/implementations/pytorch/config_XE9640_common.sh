## System config params
export DGXNGPU=4
export DGXSOCKETCORES=56
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export NCCL_SOCKET_IFNAME=^eno
