#!/bin/bash
set -x
source ~/miniconda3/bin/activate $CONDA_ENV
# setup training dataset
export PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH=$SINGLEHOT_PATH
export MATERIALIZED_DATASET_PATH=$MULTIHOT_PATH


export RANK=$1 # setup training info
export node=$2
export run=$4
scontrol show hostname $SLURM_NODELIST >./hostfile${node}_${run}
hostfile="./hostfile${node}_${run}"
#hostfile="./hostfile$node"
# export LOCAL_RANK=0
export MASTER_ADDR=0
export MASTER_PORT=1081
# export RANK=0
# export WORLD_SIZE=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export KMP_AFFINITY=granularity='fine,compact,1,0'
export KMP_BLOCKTIME=1
export MULTIHOTDATA=0
export OVERARCH_LAYER="1024,1024,512,256,1"
export DENSEARCH_LAYER="512,256,128"
export NUM_EMBEDDINGS="40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"
export INTERACTION_TYPE="dcn"
export MULTIHOT_SIZE="3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1"
#export BATCH_SIZE=65536
export TOTAL_TRAINING_SAMPLES=4195197692
export BATCH_SIZE=$3
export SEED=$(date +%s)
export DLRM_DEFAULT_ARGS="--embedding_dim 128 \
    --validation_freq_within_epoch $((TOTAL_TRAINING_SAMPLES / (BATCH_SIZE * 20)))  \
    --validation_auroc 0.80275 \
    --limit_val_batches ${BATCH_SIZE} \
    --epochs 1 \
    --pin_memory \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 0.004 \
    --adagrad \
    --seed $SEED \
    --mmap_mode \
    --over_arch_layer_sizes ${OVERARCH_LAYER} \
    --dense_arch_layer_sizes ${DENSEARCH_LAYER} \
    --num_embeddings_per_feature ${NUM_EMBEDDINGS} \
    --interaction_type=${INTERACTION_TYPE} "
#    --lr_warmup_steps=6400 \
#    --lr_decay_start=51200 \
#    --lr_decay_steps=21100 "

export CCL_LOG_LEVEL=info
export LOG_TAG=`date +"%Y-%m-%d_%H-%M-%m"`
mkdir -p rank${RANK}_node${node}_bs$BATCH_SIZE
export LOG_FILE="./rank${RANK}_node${node}_bs$BATCH_SIZE/mlperf-training-$LOG_TAG.log"
export SCRIPT=run_mlperf-$LOG_TAG.sh
cp run_mlperf.sh $SCRIPT
#       --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
#       --num_embeddings_per_feature "40000,39060,17295,7424,20265,3,7122,1543,63,40000,30679,4052,10,2209,11938,155,4,976,14,40000,40000,40000,5901,12973,108,36" \
#rm mlperf-training-latest.log
#ln -s $LOG_FILE mlperf-training-latest.log
# run dlrm mlperf with materialized dataset

source ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bind_pt-1.13.0+cpu-py3.9-linux-x86_64.egg/oneccl_bindings_for_pytorch/env/setvars.sh
#source ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/env/setvars.sh
#source /home/daisyden/dlrmv2/dlrm_0917_yang/frameworks.ai.benchmarking.mlperf.develop.training-datacenter/closed/Intel/dlrm_v2/torchrec_cpu/oneCCL/build/_install/env/setvars.sh

function nw_config()
{
    export MASTER_ADDR=$(head -1 $hostfile)
    cat $hostfile
    if [ "$RANK" = "16" ];then
          export I_MPI_PIN_DOMAIN=[0x3fffffffffffff,0x3fffffffffffff00000000000000,]
          export CCL_WORKER_AFFINITY='166,167,222,223'
          export CCL_ALLREDUCE=rabenseifner
          export CCL_GATHERV=direct
          export CCL_ALLTOALL=scatter
          export CCL_WORKER_COUNT=2
          export ppn=2
          export OMP_NUM_THREADS=54
    fi

    if [ "$RANK" = "8" ];then
          export I_MPI_PIN_DOMAIN=[0xfffffffffffff,0xfffffffffffff00000000000000,]
          export CCL_WORKER_AFFINITY='164,165,166,167,220,221,222,223'
          export CCL_ALLREDUCE=rabenseifner
          export CCL_GATHERV=direct
          export CCL_ALLTOALL=scatter
          export CCL_WORKER_COUNT=4
          export CCL_ATL_TRANSPORT=ofi
          export PSM3_IDENTIFY=1
          export ppn=1
          export OMP_NUM_THREADS=52
    fi

    if [ "$RANK" = "4" ];then
          export I_MPI_PIN_DOMAIN=[0x7fffffffffffff,0x7fffffffffffff00000000000000,]
          export CCL_WORKER_AFFINITY='167,223'
          export CCL_ALLREDUCE=rabenseifner
          export CCL_GATHERV=direct
          export CCL_ALLTOALL=scatter
          export CCL_WORKER_COUNT=1
          export CCL_ATL_TRANSPORT=ofi
	  export PSM3_IDENTIFY=1
	  export ppn=1
          export OMP_NUM_THREADS=55
    fi

    if [ "$RANK" = "2" ];then
          export I_MPI_PIN_DOMAIN=[0x7fffffffffffff,0x7fffffffffffff00000000000000,]
          export CCL_WORKER_AFFINITY='167,223'
          export CCL_ALLREDUCE=rabenseifner
          export CCL_GATHERV=direct
          export CCL_ALLTOALL=naive
          export CCL_WORKER_COUNT=1
          export ppn=1
          export OMP_NUM_THREADS=55
    fi
			
    echo I_MPI_PIN_DOMAIN=$I_MPI_PIN_DOMAIN
    echo CCL_WORKER_AFFINITY=$CCL_WORKER_AFFINITY
    echo CCL_ALLREDUCE=$CCL_ALLREDUCE
    echo CCL_ALLTOALL=$CCL_ALLTOALL
    echo CCL_WORKER_COUNT=$CCL_WORKER_COUNT
    echo ppn=$ppn
    echo OMP_NUM_THREADS=$OMP_NUM_THREADS

    export CCL_BF16=avx512bf
    #cp /usr/lib64/libfabric.so.1.18.1  ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/lib/libfabric.so.1
    #cp /usr/lib64/libfabric/libpsm3-fi.so ${CONDA_PREFIX}/lib/python3.9/site-packages/oneccl_bindings_for_pytorch/lib/prov/libpsm3-fi.so
    #export CCL_LOG_LEVEL=info

    margs="--genv CCL_WORKER_COUNT=${CCL_WORKER_COUNT}"
    margs="$margs --genv CCL_MNIC=local"
    margs="$margs --genv CCL_MNIC_COUNT=2"
    margs="$margs --genv CCL_MNIC_NAME='irdma-cvl01tf2,irdma-cvl02tf2,irdma-cvl11tf2,irdma-cvl12tf2'"
    margs="$margs --genv CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
    margs="$margs --genv PSM3_ALLOW_ROUTERS=1"
    margs="$margs --genv PSM3_RDMA=1"
    margs="$margs --genv PSM3_RV_MR_CACHE_SIZE=8192"
    margs="$margs --genv FI_PROVIDER_PATH=/usr/lib64/libfabric"
    margs="$margs --genv PSM3_NIC_SPEED=100000"
    margs="$margs --genv PSM3_KASSIST_MODE=none"
    margs="$margs --genv PSM3_NIC=irdma*"
    margs="$margs --genv PSM3_MULTI_EP=1"
    margs="$margs --genv FI_PROVIDER=psm3"
}


nw_config

#mpirun $margs --hostfile $hostfile ./run_clean.sh 
hosts=$(cat $hostfile)
for host in $hosts
do
	ssh $host bash ./run_clean.sh
done

me=$(basename "$0")
lineno=${LINENO}
python -c "
import mlperf_logging.mllog as mllog
import mlperf_logging.mllog.constants as mllog_constants
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True, metadata={ \"file\": \"$me\", \"lineno\": $lineno } ) "


if [ $MULTIHOTDATA -ge 1 ]
then
    echo 'Multi hot training for dlrm v2'
OMP_NUM_THREADS=$OMP_NUM_THREADS mpirun $margs --hostfile $hostfile -n $RANK -ppn $ppn -l python dlrm_main_mlperf.py \
           ${DLRM_DEFAULT_ARGS} \
           --synthetic_multi_hot_criteo_path ${MATERIALIZED_DATASET_PATH} \
           --dense_labels_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} 2>&1| tee $LOG_FILE
	   #--enable_profiling \
else
    #one hot
    echo 'Onehot training for dlrm v2'
OMP_NUM_THREADS=$OMP_NUM_THREADS mpirun $margs --hostfile $hostfile -n $RANK -ppn $ppn -l python dlrm_main_mlperf.py \
           ${DLRM_DEFAULT_ARGS} \
           --multi_hot_sizes=${MULTIHOT_SIZE} \
           --multi_hot_distribution_type "uniform" \
           --in_memory_binary_criteo_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} \
           --dense_labels_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} 2>&1| tee $LOG_FILE
fi

mkdir -p rank${RANK}_node${node}_bs$BATCH_SIZE
mv rank${rank}_node${node}_run*_bs${BS}.txt rank${RANK}_node${node}_bs$BATCH_SIZE/.

set +x
