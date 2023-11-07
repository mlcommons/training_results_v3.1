#!/bin/bash
#SBATCH --job-name single_stage_detector

# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

set -euxo pipefail
# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"
# Vars with defaults
: "${MLPERF_RULESET:=3.1.0}"
: "${MLPERF_CLUSTER_NAME:='unknown'}"
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${WORK_DIR:=/workspace/ssd}"
# ci automagically sets this correctly on Selene
: "${LOGDIR:=./results}"
: "${ABSLOGDIR:=${PWD}/results}"
: "${POWERCMDDIR:=' '}"
: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"
: "${SCRATCH_SPACE:="/raid/scratch"}"

# Scaleout brdige
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${NCCL_TEST:=1}"
: "${SYNTH_DATA:=0}"
: "${EPOCH_PROF:=0}"
: "${DISABLE_CG:=0}"
# API Logging defaults
: "${API_LOGGING:=0}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
# Set GPC clock for MaxQ and minEDP
: "${SET_MAXQ_CLK:=0}"
: "${SET_MINEDP_CLK:=0}"
# Monitor for debugging hangs
: "${HANG_MONITOR_TIMEOUT:=0}"
: "${ATTEMPT_CUDA_GDB_CORE_DUMP:=0}"
: "${POSTPROCESS_CUDA_GDB_CORE_DUMP:=0}"  # set to 1 to extract active kernel info from dumps.
: "${REMOVE_CUDA_GDB_CORE_DUMP:=1}"  # set to 1 to remove coredumps after processing. Will save a lot of disk space. Valid if POSTPROCESS_CUDA_GDB_CORE_DUMP is 1

# RUNSUB_DIR is the directory containing the run.sub script, so we can call
# other scripts relative to the location of the run.sub script
if [[ "${SLURM_JOB_ID}" ]]; then
    export RUNSUB_DIR=$(dirname $(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}'))
else
    export RUNSUB_DIR=$(dirname "${BASH_SOURCE[0]}")
fi

export MLPERF_SLURM_FIRSTNODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"
echo "using MLPERF_SLURM_FIRSTNODE \"${MLPERF_SLURM_FIRSTNODE}\" of list \"${SLURM_JOB_NODELIST}\""

# pyxis sometimes leaves containers lying around which can really confuse things:
cleanup_pyxis() {
    srun --ntasks="${SLURM_JOB_NUM_NODES}" /bin/bash -c 'if [[ "$(enroot list)" ]]; then enroot remove -f $(enroot list); fi'
}
trap cleanup_pyxis TERM EXIT
cleanup_pyxis

export MODEL_NAME="single_stage_detector"
export MODEL_FRAMEWORK="pytorch"
LOGBASE="${DATESTAMP}"
SPREFIX="${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"

if [ ${TIME_TAGS} -gt 0 ]; then
    LOGBASE="${SPREFIX}_mllog"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nsys"
    else
        LOGBASE="${SPREFIX}_nsys"
    fi
fi
if [ ${SYNTH_DATA} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_synth"
    else
        LOGBASE="${SPREFIX}_synth"
    fi
fi
if [ ${EPOCH_PROF} -gt 0 ]; then
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_epoch"
    else
        LOGBASE="${SPREFIX}_epoch"
    fi
fi
if [ ${DISABLE_CG} -gt 0 ]; then
    EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--cuda-graphs//')
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nocg"
    else
        LOGBASE="${SPREFIX}_nocg"
    fi
fi

# do we need to fetch the data from lustre to /raid/scratch?
srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "rm -rf ${SCRATCH_SPACE}/local-root || true"
if [[ "${LOCALDISK_FROM_SQUASHFS:-}" ]]; then
  LOCAL_SQUASHFS="${LOCALDISK_FROM_SQUASHFS}"
  srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p ${SCRATCH_SPACE}
  # LOCALDISK_FROM_SQUASHFS should be the path/name of a squashfs file on /lustre
  if [[ "${LOCALDISK_FROM_SQUASHFS}" == *lustre* ]] || [[ "${LOCALDISK_FROM_SQUASHFS}" == *mnt* ]]; then
    echo "fetching ${LOCALDISK_FROM_SQUASHFS}"
    srun --ntasks="${SLURM_JOB_NUM_NODES}" ./copy_sqsh.sh
    LOCAL_SQUASHFS=${SCRATCH_SPACE}/tmp.sqsh
  fi
  echo "unsquashing ${LOCAL_SQUASHFS}"
  time srun --ntasks="${SLURM_JOB_NUM_NODES}" unsquashfs -no-progress -dest ${SCRATCH_SPACE}/local-root "${LOCAL_SQUASHFS}"
fi

readonly LOG_FILE_BASE="${LOGDIR}/${LOGBASE}"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
_cont_mounts="${DATADIR}:/datasets/open-images-v6,${LOGDIR}:/results,${BACKBONE_DIR}:/root/.cache/torch"

# API Logging
if [ "${API_LOGGING}" -eq 1 ]; then
    API_LOG_DIR=${API_LOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${API_LOG_DIR}
    _cont_mounts="${_cont_mounts},${API_LOG_DIR}:/logs"

    # Create JSON file for cuDNN
    JSON_MODEL_NAME="MLPERF_${MODEL_NAME}_${APILOG_MODEL_NAME}_${MODEL_FRAMEWORK}_train"
    JSON_README_LINK="${README_PREFIX}/${MODEL_NAME}/${MODEL_FRAMEWORK}/README.md"
    JSON_FMT='{model_name: $mn, readme_link: $rl, configs: {($dt): [$bs]}, sweep: {($dt): [$bs]}}'
    JSON_OUTPUT="${JSON_MODEL_NAME}.cudnn.json"
    jq -n --indent 4 --arg mn $JSON_MODEL_NAME --arg rl $JSON_README_LINK --arg dt $APILOG_PRECISION --arg bs $BATCHSIZE "$JSON_FMT" > ${API_LOG_DIR}/$JSON_OUTPUT
fi
if [ "${JET:-0}" -eq 1 ]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )
srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun \
    --ntasks="${SLURM_JOB_NUM_NODES}" \
    --container-image="${CONT}" \
    --container-name="${_cont_name}" \
    true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 73698008 -e 73698008 -d half -G 1    ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

fi

#ssh to nodes for power measurements
NODELIST=$(scontrol show hostnames ${SLURM_JOB_NODELIST})
NODELIST=(${NODELIST[*]})
if [ -f "$POWERCMDDIR/power_monitor.sh"  ]; then
    ( umask 0002; mkdir -p "${ABSLOGDIR}" )
    for i in "${NODELIST[@]}"
    do
        ssh $i 'export NODENAME='"'$i'"';export ABSLOGDIR='"'$ABSLOGDIR'"';export SLURM_JOB_NODELIST='"'$SLURM_JOB_NODELIST'"';export SLURM_JOB_ID='"'$SLURM_JOB_ID'"';POWERCMDDIR='"'$POWERCMDDIR'"';bash ${POWERCMDDIR}/power_monitor.sh' &
#	break
    done
fi

if [[ "${SET_MAXQ_CLK}" == "1" ]] || [[ "${SET_MINEDP_CLK}" == "1" ]]; then
	if [[ "${SET_MAXQ_CLK}" == "1" ]]; then
		GPCCLK=${MAXQ_CLK}
	fi
	if [[ "${SET_MINEDP_CLK}" == "1" ]]; then
		GPCCLK=${MINEDP_CLK}
	fi
	for i in "${NODELIST[@]}"
	do
		ssh $i 'export GPCCLK='"'$GPCCLK'"';sudo nvidia-smi -lgc ${GPCCLK}'
	done
fi

if [ "${HANG_MONITOR_TIMEOUT-0}" -gt 0 ]; then
  HANG_MONITOR_EXEC_CMD="
    srun \
      --overlap -l --no-container-mount-home --container-mounts=${_cont_mounts} \
      --container-name=${_cont_name} --container-workdir=${WORK_DIR} --ntasks=${SLURM_JOB_NUM_NODES} \
      bash scripts/tracebacks/dump_tracebacks_node.sh"

    if [ "${ATTEMPT_CUDA_GDB_CORE_DUMP}" == "1" ]; then
      echo "Enabling user triggered CPU core dump"
      export CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1
      export CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1

      export CUDA_COREDUMP_PIPE_BASEDIR="/results/coredumps/${DATESTAMP}"
      export CUDA_COREDUMP_PIPE_HOSTDIR="${LOGDIR}/coredumps/${DATESTAMP}"
      export CUDA_COREDUMP_PIPE="${CUDA_COREDUMP_PIPE_BASEDIR}/corepipe.cuda.%h.%p"
      export CUDA_COREDUMP_FILE="${CUDA_COREDUMP_PIPE_BASEDIR}/core_%h_%p.nvcudmp"

      mkdir -p "${CUDA_COREDUMP_PIPE_HOSTDIR}"

      HANG_MONITOR_EXEC_CMD+=";
        srun \
          --overlap -l --no-container-mount-home --container-mounts=${_cont_mounts} \
          --container-name=${_cont_name} --container-workdir=${WORK_DIR} --ntasks=${SLURM_JOB_NUM_NODES} \
          bash scripts/tracebacks/dump_core_node.sh"
    fi

  source "${RUNSUB_DIR}/scripts/tracebacks/hang_monitor.sh"
  ( TRACEBACKS_ID=$DATESTAMP hang_monitor &> "${LOGDIR}/${SPREFIX}_hang_monitor.log" ) &
  hang_monitor_pid=$!
else
  hang_monitor_pid=
fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (

        echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST} ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}"
        # Print system info
        srun -N1 -n1 --container-name="${_cont_name}" python -c ""

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && ${DROPCACHE_CMD}"
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
from mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi
        sleep 30
        # Run experiment
        srun \
            --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" \
            --ntasks-per-node="${DGXNGPU}" \
            --container-name="${_cont_name}" \
            --container-mounts="${_cont_mounts}" \
            --container-workdir=${WORK_DIR} \
            slurm2pytorch ./run_and_time.sh
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
    sleep 30
    # compliance checker
    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
           --container-mounts="$(realpath ${LOGDIR}):/results"   \
           --container-workdir="/results"                        \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${LOGBASE}_${_experiment_index}.log" \
     || true
    fi

    if [ "${POSTPROCESS_CUDA_GDB_CORE_DUMP}" -eq 1 ] \
        && [ "${HANG_MONITOR_TIMEOUT-0}" -gt 0 ] \
        && [ "${ATTEMPT_CUDA_GDB_CORE_DUMP}" == "1" ] \
        && [ -n "$(ls -A ${CUDA_COREDUMP_PIPE_HOSTDIR}/*.nvcudmp)" ]; then
      echo "Postprocessing CUDA core dumps"
      srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
           --container-mounts="$(realpath ${LOGDIR}):/results"   \
           --container-workdir="${WORK_DIR}"                        \
           bash scripts/tracebacks/postprocess_core_dumps.sh     \
    || true
    fi

    if [ "${JET:-0}" -eq 1 ]; then
      JET_CREATE=${JET_CREATE:-}" --data workload.spec.nodes=${DGXNNODES} --data workload.spec.name=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --data workload.key=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --mllogger "
      srun -N1 -n1 --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" bash -c "${JET_CREATE} /results/${LOGBASE}_${_experiment_index}.log && ${JET_UPLOAD}"
    fi
done

# Kill hang monitors
if [ -n ${hang_monitor_pid} ] && ps -p $hang_monitor_pid > /dev/null; then
  pkill -P $hang_monitor_pid
fi
