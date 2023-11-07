export BATCHSIZE=48 #54
export GRADIENT_STEPS=1
export LR=0.01
export MAX_SAMPLES_TERMINATION=4500000
export MAX_STEPS=100000
export OPT_LAMB_BETA_1=0.60466
export OPT_LAMB_BETA_2=0.85437
export START_WARMUP_STEP=0
export WARMUP_PROPORTION=0.0
#export WARMUP_STEPS=0
export WEIGHT_DECAY_RATE=0.1
export INIT_LOSS_SCALE=4096.0

export SBATCH_NETWORK=sharp
#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=4 --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu "
export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=2 --fused_bias_fc --fused_bias_mha --fused_dropout_add  --fused_gemm_gelu "
export PHASE=2
export EVAL_ITER_START_SAMPLES=150000
export EVAL_ITER_SAMPLES=150000

## System run parms
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=01:15:00

## System config params
export DGXNGPU=16
#export DGXSOCKETCORES=32
#export DGXNSOCKET=2
#export DGXHT=1         # HT is on is 2, HT off is 1
#export SLURM_NTASKS=${DGXNGPU}
export CUDA_VISIBLE_DEVICES="0,1"

# System name
export MLPERF_SUBMISSION_ORG="Clemson Reseach Computing and Data"
export MLPERF_SUBMISSION_PLATFORM="${DGXSYSTEM}"
#export OMP_NUM_THREADS=2
#NCCL_SOCKET_IFNAME=mlx5_0,mlx5_1



#### DL params
#export BATCHSIZE=48
#export PACKING_FACTOR=2
#export GRADIENT_STEPS=1
#export LR=0.00096
#export MAX_SAMPLES_TERMINATION=4500000
##export MAX_STEPS=3680
#export MAX_STEPS=7360
#export OPT_LAMB_BETA_1=0.60466
#export OPT_LAMB_BETA_2=0.85437
#export START_WARMUP_STEP=0
#export WARMUP_PROPORTION=0.0
#export WEIGHT_DECAY_RATE=0.1
#export INIT_LOSS_SCALE=1024.0
##
#export EXTRA_PARAMS="--dense_seq_output --pad_fmha --fused_bias_fc --fused_bias_mha --fused_dropout_add --fused_gemm_gelu --packed_samples --use_transformer_engine2 "
#export PHASE=2
#export EVAL_ITER_START_SAMPLES=150000
#export EVAL_ITER_SAMPLES=150000
#
### force to use packed trainset
#export DATADIR_PHASE2=${DATADIR_PHASE2_PACKED}
#
## DL params
#export BATCHSIZE=112
#export GRADIENT_STEPS=2
#export LR=3.5e-4
#export MAX_SAMPLES_TERMINATION=4500000
#export MAX_STEPS=8041
#export OPT_LAMB_BETA_1=0.9
#export OPT_LAMB_BETA_2=0.999
#export START_WARMUP_STEP=0
#export WARMUP_PROPORTION=0.0
#export WEIGHT_DECAY_RATE=0.01
#export INIT_LOSS_SCALE=1024.0

#export EXTRA_PARAMS="--dense_seq_output --unpad --unpad_fmha --exchange_padding --dwu-group-size=4 "

#export PHASE=2
#export EVAL_ITER_START_SAMPLES=150000
#export EVAL_ITER_SAMPLES=150000

