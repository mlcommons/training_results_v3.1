## System config params
export DGXNGPU=8
export DGXSOCKETCORES=48
export DGXNSOCKET=2
export DGXHT=1         # HT is on is 2, HT off is 1
export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
