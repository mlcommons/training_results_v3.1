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

set +x
set -e

# Only rank print
[ "${SLURM_LOCALID-0}" -ne 0 ] && set +x


# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# Set variables
[ "${DEBUG}" = "1" ] && set -x
LR=${LR:-0.0001}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
BATCHSIZE=${BATCHSIZE:-2}
EVALBATCHSIZE=${EVALBATCHSIZE:-${BATCHSIZE}}
NUMEPOCHS=${NUMEPOCHS:-10}
LOG_INTERVAL=${LOG_INTERVAL:-20}
DATASET_DIR=${DATASET_DIR:-"/datasets/open-images-v6"}
TORCH_HOME=${TORCH_HOME:-"/torch-home"}
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NSYS_FLAG=${NSYS_FLAG:-0}
NCU_FLAG=${NCU_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
DISABLE_CG=${DISABLE_CG:-0}
ENABLE_IB_BINDING=${ENABLE_IB_BINDING:-1}

echo "RANK ${RANK-}: LOCAL_RANK ${LOCAL_RANK-}, MASTER_ADDR ${MASTER_ADDR-}, MASTER_PORT ${MASTER_PORT-}, WORLD_SIZE ${WORLD_SIZE-}, MLPERF_SLURM_FIRSTNODE ${MLPERF_SLURM_FIRSTNODE-}, SLURM_JOB_ID ${SLURM_JOB_ID-}, SLURM_NTASKS ${SLURM_NTASKS-}, SLURM_PROCID ${SLURM_PROCID-}, SLURM_LOCALID ${SLURM_LOCALID-}, OMP_NUM_THREADS ${OMP_NUM_THREADS-}"

# run benchmark
echo "running benchmark"
DATESTAMP=${DATESTAMP:-$(date +'%y%m%d%H%M%S%N')}
if [[ ${NVTX_FLAG} -gt 0 || ${NSYS_FLAG} -gt 0 ]]; then
  NSYS_OUT="/results/single_stage_detector_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${SLURM_PROCID}_${SYNTH_DATA}_${DISABLE_CG}_${DATESTAMP}.nsys-rep"
  PRE_CMD=('nsys' 'profile' '--capture-range' 'cudaProfilerApi' '--capture-range-end' 'stop' '--sample=none' '--cpuctxsw=none' '--trace=cuda,nvtx' '--stats' 'true' '--cuda-graph-trace=node' '-f' 'true' '-o' ${NSYS_OUT})
elif [ ${NCU_FLAG} -gt 0 ]; then
  NCU_OUT="/results/single_stage_detector_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${SLURM_PROCID}_${SYNTH_DATA}_${DISABLE_CG}_${DATESTAMP}.ncu-rep"
  PRE_CMD=('ncu' '--target-processes=all' '--profile-from-start=no' '--graph-profiling=node' '--cache-control=none' '--replay-mode=application' '--metrics=sm__warps_active.avg.pct_of_peak_sustained_active' '-o' ${NCU_OUT})
else
  PRE_CMD=()
fi

if [ ${SYNTH_DATA} -gt 0 ]; then
EXTRA_PARAMS+=" --syn-dataset --cuda-graphs-syn "
EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--dali//')
fi

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    IB_BIND=''
    if [[ "${SLURM_JOB_NUM_NODES}" -gt 1 && "${ENABLE_IB_BINDING}" -eq 1 ]]; then
      IB_BIND='--ib=single'
    fi
    CMD=( 'bindpcie' ${IB_BIND} '--' ${PRE_CMD[@]} 'python' '-u' )
  else
    CMD=( ${PRE_CMD[@]} 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker, we've been launched with `torch_run`
  CMD=( ${PRE_CMD[@]} "python" )
fi


if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # Collect logs only on a single rank as a workaround for a NCCL logging bug
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

PARAMS=(
      --lr                      "${LR}"
      --batch-size              "${BATCHSIZE}"
      --eval-batch-size         "${EVALBATCHSIZE}"
      --epochs                  "${NUMEPOCHS}"
      --print-freq              "${LOG_INTERVAL}"
      --dataset-path            "${DATASET_DIR}"
      --warmup-epochs           "${WARMUP_EPOCHS}"
)

# run training
${LOGGER:-} "${CMD[@]}" train.py "${PARAMS[@]}" ${EXTRA_PARAMS} ; ret_code=$?

set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="SINGLE_STAGE_DETECTOR"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"
