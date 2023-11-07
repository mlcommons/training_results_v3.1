## DL params
export BATCHSIZE=48
export GRADIENT_STEPS=1
export LR=0.002
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=2254
export OPT_LAMB_BETA_1=0.66
export OPT_LAMB_BETA_2=0.996
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
export WEIGHT_DECAY_RATE=0.01
export INIT_LOSS_SCALE=4096.0


#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=2 --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "
export EXTRA_PARAMS="--dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu --packed_samples --cuda_graph_mode 'segmented' "
export PHASE=2
export EVAL_ITER_START_SAMPLES=175000
export EVAL_ITER_SAMPLES=175000

## System run parms
export DGXNNODES=18
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
export DGXNGPU=2
#export DGXSOCKETCORES=32
#export DGXNSOCKET=2
#export DGXHT=1         # HT is on is 2, HT off is 1
#export SLURM_NTASKS=${DGXNGPU}

# System name
export MLPERF_SUBMISSION_ORG="Clemson Research Computing and Data"
export MLPERF_SUBMISSION_PLATFORM="${DGXSYSTEM}"
export OMP_NUM_THREADS=8
#NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1
#Dual IB
#NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1
#single IB
#NCCL_SOCKET_IFNAME=mlx5_0
#single OPA
NCCL_SOCKET_IFNAME=hfi1_0
#Dual OPA
#NCCL_SOCKET_IFNAME=hfi1_0,hfi1_1
