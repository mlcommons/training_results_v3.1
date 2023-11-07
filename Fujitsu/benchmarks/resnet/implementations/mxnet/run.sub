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

#SBATCH --job-name image_classification
set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${MLPERF_RULESET:=3.0.0}"
: "${MLPERF_CLUSTER_NAME:='unknown'}"
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${LOGDIR:=./results}"
: "${ABSLOGDIR:=${PWD}/results}"
: "${POWERCMDDIR:=' '}"
: "${MEASURE_POWER:=0}"
: "${COPY_DATASET:=}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${DROPCACHE_CMD:="sudo /sbin/sysctl vm.drop_caches=3"}"
: "${SET_MAXQ_CLK:=0}"
: "${SET_MINEDP_CLK:=0}"

TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-1}
SYNTH_DATA=${SYNTH_DATA:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
DISABLE_CG=${DISABLE_CG:-0}

# pyxis sometimes leaves containers lying around which can really confuse things:
cleanup_pyxis() {
    srun --ntasks="${SLURM_JOB_NUM_NODES}" /bin/bash -c 'if [[ "$(enroot list)" ]]; then enroot remove -f $(enroot list); fi'
}
trap cleanup_pyxis TERM EXIT
cleanup_pyxis

if [[ "${SLURM_JOB_ID}" ]]; then
    export RUNSUB_DIR=$(dirname $(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/Command=/{print $2}'))
else
    export RUNSUB_DIR=$(dirname "${BASH_SOURCE[0]}")
fi

# do we need to fetch the data from remote disk into local /tmp disk?
if [[ "${MNT_TARFILE_PATH:-}" ]]; then
    # make sure we didn't accidentally specify the remote disk as the tmpdir
    if [[ "${TMP_ROOT_PATH}" == *mnt* ]]; then
	echo "ERROR: ${TMP_ROOT_PATH} looks like a lustre mount rather than a tmp dir, yet TARFILE_FOR_PREPROC_DATA is set to ${TARFILE_FOR_PREPROC_DATA}!!!"
	exit 1
    fi
    # manage data in tmpdir on every node
    srun --ntasks="${SLURM_JOB_NUM_NODES}" \
	 "${RUNSUB_DIR}/manage-tmp-data" \
	 "${MNT_TARFILE_PATH}" "${TMP_ROOT_PATH}"   \
	 "${TMP_MD5SUM}"
	 export DATADIR=${DATADIR_TMP}
fi

echo $COPY_DATASET
if [ ! -z $COPY_DATASET ]; then
  readonly copy_datadir=$COPY_DATASET
  srun --ntasks-per-node=1 mkdir -p "${DATADIR}"
  srun --ntasks-per-node=1 ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
  srun --ntasks-per-node=1 bash -c "ls ${DATADIR}"
fi

export MODEL_NAME="image_classification"
export MODEL_FRAMEWORK="mxnet"
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
    if [[ "$LOGBASE" == *'_'* ]];then
        LOGBASE="${LOGBASE}_nocg"
    else
        LOGBASE="${SPREFIX}_nocg"
    fi
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _logfile_base="${LOGDIR}/${LOGBASE}"
readonly _cont_name="${MODEL_NAME}_${SLURM_JOB_ID}"
_cont_mounts="${DATADIR}:/data,${LOGDIR}:/results"
if [ "${API_LOGGING:-}" -eq 1 ]; then
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


# MLPerf vars
MLPERF_HOST_OS=$(srun -N1 -n1 bash <<EOF
    source /etc/os-release
    source /etc/dgx-release || true
    echo "\${PRETTY_NAME} / \${DGX_PRETTY_NAME:-???} \${DGX_OTA_VERSION:-\${DGX_SWBUILD_VERSION:-???}}"
EOF
)
export MLPERF_HOST_OS

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )
srun --ntasks="${SLURM_JOB_NUM_NODES}" mkdir -p "${LOGDIR}"

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
srun --ntasks="${SLURM_JOB_NUM_NODES}" --container-image="${CONT}" --container-name="${_cont_name}" true
srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

#Run NCCL B/W Test
#https://github.com/nvidia/nccl-tests
echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 51.2M -e 51.2M -d half -G 1    ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

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


# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}  ${MLPERF_CLUSTER_NAME} ${DGXSYSTEM}"
        # Print system info
        srun -N1 -n1 --container-name="${_cont_name}" python -c "
from mlperf_log_utils import mllogger
mllogger.mlperf_submission_log(mllogger.constants.RESNET)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --mpi="${SLURM_MPI_TYPE:-pmix}" bash -c "echo -n 'Clearing cache on ' && hostname && sync && ${DROPCACHE_CMD}"
            srun --ntasks="${SLURM_JOB_NUM_NODES}" --mpi="${SLURM_MPI_TYPE:-pmix}" --container-name="${_cont_name}" python -c "
from mlperf_log_utils import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
	srun --kill-on-bad-exit=0 --mpi="${SLURM_MPI_TYPE:-pmix}" --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
            --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" \
            ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
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

  if [ "${JET:-0}" -eq 1 ]; then
    JET_CREATE=${JET_CREATE:-}" --data workload.spec.nodes=${DGXNNODES} --data workload.spec.name=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --data workload.key=${MODEL_NAME}_${MODEL_FRAMEWORK}_${DGXSYSTEM} --mllogger "
    srun -N1 -n1 --container-name="${_cont_name}" --container-mounts="${_cont_mounts}" bash -c "${JET_CREATE} /results/${LOGBASE}_${_experiment_index}.log && ${JET_UPLOAD}"
  fi

done
