#!/bin/bash

# setup training dataset
export PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH=/data/dlrm_2_dataset/one_hot/
export MATERIALIZED_DATASET_PATH=/data/dlrm_2_dataset/multi_hot/

# setup training info
export LOCAL_RANK=0
export MASTER_ADDR=0
export MASTER_PORT=1081
export RANK=0
export WORLD_SIZE=1

# run dlrm mlperf with materialized dataset
python dlrm_main.py \
       --embedding_dim 128 \
       --num_embeddings 40000000 \
       --over_arch_layer_sizes "1024,1024,512,256,1" \
       --dense_arch_layer_sizes "512,256,128" \
       --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36" \
       --validation_freq_within_epoch 80 \
       --limit_val_batches 2048 \
       --epochs 1 \
       --pin_memory \
       --batch_size 65536 \
       --learning_rate 15.0 \
       --seed 0 \
       --mmap_mode \
       --interaction_type="dcn" \
       --synthetic_multi_hot_criteo_path $MATERIALIZED_DATASET_PATH \
       --dense_labels_path $PREPROCESSED_CRITEO_1TB_CLICK_LOGS_DATASET_PATH
