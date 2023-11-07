#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HOME=/hf_home

export HYDRA_FULL_ERROR=1

export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
#RANDOM_SEED=13977
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=19002
#export WORLD_SIZE=${SLURM_NTASKS}
#export WORLD_SIZE=8
export WORLD_SIZE=${DGXNGPU}
echo "WORLD_SIZE=$WORLD_SIZE"

export SLURM_PROCID=${OMPI_COMM_WORLD_LOCAL_RANK}
export SLURM_NTASKS_PER_NODE=${DGXNGPU}
export SLURM_NTASKS=${DGXNGPU}
export SLURM_NODEID=0
export SLURM_LOCALID=${SLURM_PROCID}

readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
#echo "LOCAL_RANK=$LOCAL_RANK"
#echo "local_rank=$local_rank"
#echo "OMPI_COMM_WORLD_LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK"


start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

: "${DGXNNODES:?DGXNNODES must be set}"
: "${DGXNGPU:?DGXNGPU must be set}"
: "${BATCHSIZE:?BATCHSIZE must be set}"
: "${LEARNING_RATE:?LEARNING_RATE must be set}"

#: "${EXP_NAME:=stable-diffusion2-train-$(date +%y%m%d%H%M%S%N)}"
: "${DATETIME:=$(date +%y%m%d%H%M%S%N)}"
: "${EXP_NAME:=stable-diffusion2-train-$DATETIME}"

: "${CONFIG_PATH:=conf}"
: "${CONFIG_NAME:=sd2_mlperf_train_moments}"
: "${CONFIG_MAX_STEPS:=1000}"
: "${RANDOM_SEED:=$RANDOM}"
: "${FLASH_ATTENTION:=True}"
: "${CHECKPOINT_STEPS:=1000}"
: "${INFER_NUM_IMAGES:=30000}"

# CLEAR YOUR CACHE HERE
python -c "
from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger
mllogger.event(key=constants.CACHE_CLEAR, value=True)"

declare -a CMD

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 && "${ENABLE_IB_BINDING:-}" == "1" ]]; then
    IB_BIND='--ib=single'
fi

CPU_EXCLUSIVE=''
if [[ "${ENABLE_CPU_EXCLUSIVE:-1}" == "1" ]]; then
    CPU_EXCLUSIVE='--cpu=exclusive'
fi

if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    CMD=( 'bindpcie' ${CPU_EXCLUSIVE} ${IB_BIND} '--' 'python' '-u')
else
    # docker or single gpu, no need to bind
    CMD=( 'python' '-u' )
fi

CMD=( 'python' '-u' )

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
    echo "###########################################################################################!!!"
  else
    LOGGER=""
  fi
fi

# Assert $RANDOM is usable
if [ -z "$RANDOM" ]; then
    echo "RANDOM is not set!" >&2
    exit 1
fi

echo "RANDOM_SEED=${RANDOM_SEED}"
mkdir -p "/tmp/nemologs"
#added by frank
rm -rf /tmp/nemologs/*

${LOGGER:-} ${CMD[@]} main.py \
    "trainer.num_nodes=${DGXNNODES}" \
    "trainer.devices=${DGXNGPU}" \
    "trainer.max_steps=${CONFIG_MAX_STEPS}" \
    "model.optim.lr=${LEARNING_RATE}" \
    "model.optim.sched.warmup_steps=1000" \
    "model.micro_batch_size=${BATCHSIZE}" \
    "model.global_batch_size=$((DGXNGPU * DGXNNODES * BATCHSIZE))" \
    "model.unet_config.use_flash_attention=${FLASH_ATTENTION}" \
    "exp_manager.exp_dir=/tmp/nemologs" \
    "exp_manager.checkpoint_callback_params.every_n_train_steps=${CHECKPOINT_STEPS}" \
    "name=${EXP_NAME}" \
    "model.seed=${RANDOM_SEED}" \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" || exit 1

# Move checkpoints to /nemologs but only on rank 0
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Moving checkpoints to nemologs"
    cp -r /tmp/nemologs/* /nemologs
fi
python barrier.py

CKPT_PATH="/nemologs/${EXP_NAME}/checkpoints/"
echo "CKPT_PATH=${CKPT_PATH}"

# Get a list of checkpoint files ending with "*0.ckpt"
CHECKPOINTS=($(ls "${CKPT_PATH}" | grep ".*0.ckpt"))

# Print the list of checkpoints
for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "Found checkpoint: ${checkpoint}"
done

# Create `inference` directory if it doesn't exists
mkdir -p "/nemologs/${EXP_NAME}/inference"
mkdir -p "/results/${EXP_NAME}/inference"

# For running commands only for rank = 0
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

# Convert annotations from CSV to a directory of .txt
COCO_IMAGES_DIR="/datasets/coco2014/val2014_512x512_30k"
COCO_ACTIVATIONS_DIR="/datasets/coco2014"
COCO_PROMPTS_DIR=$(python evaluation/coco_caption_conversion.py --captions-tsv "/datasets/coco2014/val2014_30k.tsv")
echo "COCO_PROMPTS_DIR=${COCO_PROMPTS_DIR}"

num_images=$(ls -1 "$COCO_IMAGES_DIR" | wc -l)
echo "Number of images: ${num_images}"

num_captions=$(ls -1 "$COCO_PROMPTS_DIR" | wc -l)
echo "Number of captions: ${num_captions}"


# Loop through the checkpoints and run inference
for checkpoint in "${CHECKPOINTS[@]}"; do
    IMAGES_DEST="/nemologs/${EXP_NAME}/inference/${checkpoint}"
    echo "Running inference on checkpoint: ${checkpoint}"
    echo "Saving images to: ${IMAGES_DEST}"

    python sd_infer.py \
        "name=${EXP_NAME}" \
        "infer.out_path='${IMAGES_DEST}'" \
        "trainer.num_nodes=${DGXNNODES}" \
        "trainer.devices=${DGXNGPU}" \
        "model.restore_from_path='${CKPT_PATH}${checkpoint}'" \
        "custom.prompts_dir='${COCO_PROMPTS_DIR}'" \
        "custom.num_prompts=${INFER_NUM_IMAGES}" \
        --config-path "${CONFIG_PATH}" \
        --config-name="sd_mlperf_infer"
    echo "Done running inference on checkpoint: ${checkpoint}"
done

INFER_DEST="/nemologs/${EXP_NAME}/inference"

python -m evaluation.eval_fid \
  --coco_images_path "${COCO_IMAGES_DIR}" \
  --coco_activations_dir "${COCO_ACTIVATIONS_DIR}" \
  --fid_images_path "${INFER_DEST}" \
  --output_path "/results/${EXP_NAME}/fid.csv"

python -m evaluation.compute_clip_score \
  --captions_path "${COCO_PROMPTS_DIR}" \
  --fid_images_path "${INFER_DEST}" \
  --output_path "/results/${EXP_NAME}/clip.csv" \
  --cache_dir "/checkpoints/clip"

python report_end.py --fid "/results/${EXP_NAME}/fid.csv" --clip "/results/${EXP_NAME}/clip.csv"

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# runtime
runtime=$(( $end - $start ))
result_name="stable_diffusion"

echo "RESULT,$result_name,$runtime,$USER,$start_fmt"
