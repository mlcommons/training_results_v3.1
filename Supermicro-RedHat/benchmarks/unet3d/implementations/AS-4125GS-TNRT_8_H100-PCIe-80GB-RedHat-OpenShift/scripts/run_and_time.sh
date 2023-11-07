#!/bin/bash

# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
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

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

set -e

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
[ "${DEBUG}" = "1" ] && set -x
echo "running benchmark"

DATADIR="data"
SEED=${1:--1}
OPTIMIZER=${OPTIMIZER:-"nag"}
BATCHSIZE=${BATCHSIZE:-4}
VAL_BATCHSIZE=${VAL_BATCHSIZE:-4}
LR=${LR:-"4.0"}
MAX_EPOCHS=${MAX_EPOCHS:-4000}
LR_WARMUP_EPOCHS=${LR_WARMUP_EPOCHS:-1000}
QUALITY_THRESHOLD=${QUALITY_THRESHOLD:-"0.908"}
START_EVAL_AT=${START_EVAL_AT:-1000}
EVALUATE_EVERY=${EVALUATE_EVERY:-20}
TARGET_DIR=${TARGET_DIR:-""}
ASYNC_PARAMS=${ASYNC_PARAMS:-""}
PROFILING_PREFIX=${PROFILING_PREFIX:-""}
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
DISABLE_CG=${DISABLE_CG:-0}
ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

echo "SYNTH_DATA = ${SYNTH_DATA}"

#disable nsight when synth is on because it segment faults
#if [ ${SYNTH_DATA} -gt 0 ]; then
#NVTX_FLAG=0
#fi

if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --capture-range cudaProfilerApi --capture-range-end stop --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/image_segmentation_mxnet_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}_${DISABLE_CG}.nsys-rep "
else
 NSYSCMD=""
fi

if [ ${NVTX_FLAG--1} -gt 0 ] ||  [ ${TIME_TAGS--1} -gt 0 ]; then
MAX_EPOCHS=3
WALLTIME=$(($WALLTIME + 15))
fi

if [ ${SYNTH_DATA--1} -gt 0 ]; then
EXTRA_PARAMS+=" --loader synthetic"
fi

IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 && "${ENABLE_IB_BINDING}" -eq 1 ]]; then
  IB_BIND='--ib=single'
fi
BIND="numactl --interleave=all"
#"bindpcie --cpu=exclusive ${IB_BIND} --"

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
  else
    LOGGER=""
  fi
fi

# run training
${LOGGER:-} ${BIND} ${NSYSCMD} python -u main.py \
    --data_dir "${DATADIR}" \
    --epochs "${MAX_EPOCHS}" \
    --quality_threshold "${QUALITY_THRESHOLD}" \
    --batch_size "${BATCHSIZE}" \
    --evaluate_every "${EVALUATE_EVERY}" \
    --start_eval_at "${START_EVAL_AT}" \
    --lr_warmup_epochs "${LR_WARMUP_EPOCHS}" \
    --optimizer "${OPTIMIZER}" \
    --learning_rate "${LR}" \
    ${PRECISION} --warmup \
    --val_batch_size "${VAL_BATCHSIZE}" ${EXTRA_PARAMS} ${ASYNC_PARAMS} \
    --num_workers "${NUM_WORKERS}" \
    --input_batch_multiplier "${INPUT_BATCH_MULTIPLIER}"; ret_code=$?

set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="image_segmentation"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
