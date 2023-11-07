#!/bin/bash

# setup training dataset
export PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH=$SINGLEHOT_PATH
export MATERIALIZED_DATASET_PATH=$MULTIHOT_PATH

# setup training info
export LOCAL_RANK=0
export MASTER_ADDR=0
export MASTER_PORT=1081
export RANK=0
export WORLD_SIZE=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so
export KMP_AFFINITY=granularity='fine,compact,1,0'
export KMP_BLOCKTIME=1
#export ONECCL_PATH=/nfs/taran/train/anaconda3/envs/dlrm_v2/lib/python3.9/site-packages/oneccl_bindings_for_pytorch
#source ${ONECCL_PATH}/env/setvars.sh
export MULTIHOTDATA=0
export OVERARCH_LAYER="1024,1024,512,256,1"
export DENSEARCH_LAYER="512,256,128"
export NUM_EMBEDDINGS="40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"
export INTERACTION_TYPE="dcn"
export MULTIHOT_SIZE="3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1"
export BATCH_SIZE=65536
export DLRM_DEFAULT_ARGS="--embedding_dim 128 \
    --validation_freq_within_epoch 1600 \
    --limit_val_batches ${BATCH_SIZE} \
    --epochs 1 \
    --pin_memory \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 0.004 \
    --adagrad \
    --seed 1234 \
    --mmap_mode \
    --over_arch_layer_sizes ${OVERARCH_LAYER} \
    --dense_arch_layer_sizes ${DENSEARCH_LAYER} \
    --num_embeddings_per_feature ${NUM_EMBEDDINGS} \
    --interaction_type=${INTERACTION_TYPE} "
#    --lr_warmup_steps=6400 \
#    --lr_decay_start=51200 \
#    --lr_decay_steps=21100 "
export LOG_TAG=`date +"%Y-%m-%d_%H-%M-%m"`
export LOG_FILE=mlperf-training-$LOG_TAG.log
export SCRIPT=run_mlperf-$LOG_TAG.sh
cp run_mlperf.sh $SCRIPT
#       --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
#       --num_embeddings_per_feature "40000,39060,17295,7424,20265,3,7122,1543,63,40000,30679,4052,10,2209,11938,155,4,976,14,40000,40000,40000,5901,12973,108,36" \
rm mlperf-training-latest.log
ln -s $LOG_FILE mlperf-training-latest.log
# run dlrm mlperf with materialized dataset
if [ $MULTIHOTDATA -ge 1 ]
then
    echo 'Multi hot training for dlrm v2'
    python -u dlrm_main_mlperf.py \
           ${DLRM_DEFAULT_ARGS} \
	   --synthetic_multi_hot_criteo_path ${MATERIALIZED_DATASET_PATH} \
           --dense_labels_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} 2>&1| tee $LOG_FILE
else
    #one hot
    echo 'Onehot training for dlrm v2'
    python -u dlrm_main_mlperf.py \
     	    ${DLRM_DEFAULT_ARGS} \
           --multi_hot_sizes=${MULTIHOT_SIZE} \
           --multi_hot_distribution_type "uniform" \
           --in_memory_binary_criteo_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} \
           --dense_labels_path ${PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH} 2>&1| tee $LOG_FILE
fi
    
