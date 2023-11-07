#!/bin/bash
#SBATCH --exclusive
#SBATCH --mem=0

# Copyright (c) 2019-2023 NVIDIA CORPORATION. All rights reserved.
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

set -eux

# Vars without defaults
: "${CONT:?CONT not set}"
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${NEXP:?NEXP not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.1.0}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${API_LOGGING:=0}"
: "${CLEAR_CACHES:=1}"
: "${CONT_FILE:=/lustre/fsw/containers/${SLURM_JOBID}_$(basename ${CONT}).squashfs}"
: "${CONTAINER_PRELOAD_LUSTRE:=0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CHECK_COMPLIANCE:=1}"
: "${LOGDIR:=./results}"
: "${ABSLOGDIR:=${PWD}/results}"
: "${POWERCMDDIR:=' '}"
: "${NSYSCMD:=""}"
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${NCCL_TEST:=1}"
: "${SYNTH_DATA:=0}"
: "${EPOCH_PROF:=0}"
: "${DISABLE_CG:=0}"
: "${WORK_DIR:=/workspace/bert}"


export MLPERF_SLURM_FIRSTNODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST-}" | head -n1)"

# pyxis sometimes leaves containers lying around which can really confuse things:
cleanup_pyxis() {
    srun --ntasks="${SLURM_JOB_NUM_NODES}" /bin/bash -c 'if [[ "$(enroot list)" ]]; then enroot remove -f $(enroot list); fi'
}
cleanup_pyxis

export MODEL_NAME="language_model"
export MODEL_FRAMEWORK="pytorch"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
LOG_BASE="${DATESTAMP}"
SPREFIX="${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"

if [ ${TIME_TAGS} -gt 0 ]; then
    LOG_BASE="${SPREFIX}_mllog"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
    if [[ "$LOG_BASE" == *'_'* ]];then
        LOG_BASE="${LOG_BASE}_nsys"
    else
        LOG_BASE="${SPREFIX}_nsys"
    fi

    if [[ ! -d "${NVMLPERF_NSIGHT_LOCATION}" ]]; then
	echo "$NVMLPERF_NSIGHT_LOCATION doesn't exist on this system!" 1>&2
	exit 1
    fi
fi
if [ ${SYNTH_DATA} -gt 0 ]; then
    if [[ "$LOG_BASE" == *'_'* ]];then
        LOG_BASE="${LOG_BASE}_synth"
    else
        LOG_BASE="${SPREFIX}_synth"
    fi
fi
if [ ${EPOCH_PROF} -gt 0 ]; then
    if [[ "$LOG_BASE" == *'_'* ]];then
        LOG_BASE="${LOG_BASE}_epoch"
    else
        LOG_BASE="${SPREFIX}_epoch"
    fi
fi
if [ ${DISABLE_CG} -gt 0 ]; then
    EXTRA_PARAMS=$(echo $EXTRA_PARAMS | sed 's/--use_cuda_graph//')
    if [[ "$LOG_BASE" == *'_'* ]];then
        LOG_BASE="${LOG_BASE}_nocg"
    else
        LOG_BASE="${SPREFIX}_nocg"
    fi
fi

if [ ${NVTX_FLAG--1} -gt 0 ] ||  [ ${TIME_TAGS--1} -gt 0 ]; then
export MAX_STEPS=100
fi

readonly LOG_FILE_BASE="${LOGDIR}/${LOG_BASE}"


#########################################################################
# preloaded squashfs option
#########################################################################

#########################################################################
# make sure "preload" tmp containers get cleaned on all possible exits (except
# kill -9)
#########################################################################
cleanup_preload_lustre() {
    if [[ "${CONTAINER_PRELOAD_LUSTRE:-0}" != "0" ]]; then
	srun --ntasks=1 rm "${CONT_FILE:?ERROR!CONT_FILE!UNDEFINED}"
    fi
}

cleanup_containers() {
    cleanup_pyxis
    cleanup_preload_lustre
}
trap cleanup_containers TERM EXIT

#########################################################################
# container preload option
#########################################################################
if [[ $CONTAINER_PRELOAD_LUSTRE -gt 0 ]]; then
    CONT_FILE="/lustre/fsw/containers/${SLURM_JOBID}_$(basename ${CONT}).squashfs"
    # Prepull container to LUSTRE
    srun --ntasks=1 enroot import --output ${CONT_FILE} docker://${CONT}
else
    CONT_FILE=${CONT}
fi

echo "CI directory structure\n"
echo $(ls)

CONT_MOUNTS="\
${DATADIR_PHASE2}:/workspace/data_phase2,\
${CHECKPOINTDIR_PHASE1}:/workspace/phase1,\
${EVALDIR}:/workspace/evaldata,\
${LOGDIR}:/results"

if [[ "${NVTX_FLAG}" -gt 0 ]]; then
    CONT_MOUNTS="${CONT_MOUNTS},${NVMLPERF_NSIGHT_LOCATION}:/nsight"
fi
if [ "${API_LOGGING}" -eq 1 ]; then
    API_LOG_DIR=${API_LOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${API_LOG_DIR}
    CONT_MOUNTS="${CONT_MOUNTS},${API_LOG_DIR}:/logs"
fi

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun --ntasks="$((SLURM_JOB_NUM_NODES))" --container-image="${CONT_FILE}" --container-name="${_cont_name}" true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

# Run NCCL test (700 MB FP16 allreduce)
#srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#        --container-image=${CONT_FILE} \
#        all_reduce_perf_mpi -b 85M -e 680M -f 2 -d half

echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 21M -e 672M -d half -G 1 -f 2 ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

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

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun --ntasks="${SLURM_JOB_NUM_NODES}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-name="${_cont_name}" python -c "
from mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
	srun -l --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
	     --container-name="${_cont_name}" --container-mounts="${CONT_MOUNTS}" \
	     --container-workdir=${WORK_DIR} \
	     slurm2pytorch "./run_and_time.sh"
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
    # compliance checker
    if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
           --container-mounts="$(realpath ${LOGDIR}):/results"   \
           --container-workdir="/results"                        \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${LOG_BASE}_${_experiment_index}.log" \
    || true
    fi
done

# Cleanup: performed by cleanup_containers (see above) on EXIT trap
