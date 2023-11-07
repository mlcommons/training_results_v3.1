#!/bin/sh
#SBATCH --job-name bert-18
#SBATCH -N 18                     # number of nodes
#SBATCH -n 36                    # total number of processes
#SBATCH --ntasks-per-node 2      # tasks per node
#SBATCH --cpus-per-task=32
#SBATCH -t 12:00:00             # wall time
#SBATCH --exclusive             # exclusive node access
#SBATCH --mem=0                 # all mem avail
#SBATCH --gres=gpu:2


module purge
#module avail
#module load shared openmpi/4.1.1
module load openmpi/4.1.5

#set -ex 

#GPU_TYPE=$( nvidia-smi -L | awk '{ print $4 }'| head -n 1 )
GPU_TYPE=`echo $(nvidia-smi -L |cut -d":" -f2 | cut -d"(" -f1 | head -n 1 |sed "s/NVIDIA//g") | xargs echo -n | sed "s/ /-/g"`

#source config_${SLURM_JOB_NUM_NODES}xXE8545x4${GPU_TYPE}.sh
#echo source config_${SLURM_JOB_NUM_NODES}xXE8545x4${GPU_TYPE}.sh
#source ./config_1xR750xax2A100-80GB-PCIe.sh
source ./configs/packed_config_${SLURM_JOB_NUM_NODES}xR750xax2A100-80GB-PCIe.sh

# Copyright (c) 2019-2022 NVIDIA CORPORATION. All rights reserved.
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

#set -eux
CONT="/scratch/nnisbet/bert.sif"
: "${LOGDIR:=/project/rcde/nnisbet/mlperf_hpc/submissions/training/Clemson/results/output}}"
: "${NEXP:=10}"
: ${DATAPATH:="/project/rcde/mlperf_data/bert/"}
export DATADIR="$DATAPATH/packed_data_hdf5"
export EVALDIR="$DATAPATH/hdf5/eval_varlength/"
export DATADIR_PHASE2="$DATADIR"
export CHECKPOINTDIR_PHASE1="$DATAPATH/phase1/"
export CHECKPOINTDIR="$CHECKPOINTDIR_PHASE1"

#######################################
# Print mount locations from file.
# Arguments:
#   arg1 - path to file containing volume mounting points
# Returns:
#   String containing comma-separated mount pairs list
#######################################
func_get_container_mounts() {
  declare -a mount_array
  readarray -t mount_array <<<$(egrep -v '^#' "${1}")
  local cont_mounts=$(envsubst <<< $(printf '%s,' "${mount_array[@]}" | sed 's/[,]*$//'))
  echo $cont_mounts
}

#######################################
# CI does not make the current directory the model directory. It is two levels up, which is different than a command line launch.
# This function looks in ${PWD} and then two levels down for a file, and updates the path if necessary.
# Arguments:
#   arg1 - expected path to file
#   arg2 - model path (e.g., language_model/pytorch/)
# Returns:
#   String containing updated (or original) path
#######################################
func_update_file_path_for_ci() {
  declare new_path
  if [ -f "${1}" ]; then
    new_path="${1}"
  else
    new_path="${2}/${1}"
  fi

  if [ ! -f "${new_path}" ]; then
    echo "File not found: ${1}"
    exit 1
  fi

  echo "${new_path}"
}

# Vars without defaults
: "${CONT:?CONT not set}"
: "${DGXSYSTEM:?DGXSYSTEM not set}"
#: "${NEXP:?NEXP not set}"

# Vars with defaults
: "${MLPERF_RULESET:=2.1.0}"
: "${API_LOG_DIR:=./api_logs}" # apiLog.sh output dir
: "${API_LOGGING:=0}"
: "${CLEAR_CACHES:=1}"
#: "${CONT_FILE:=/lustre/fsw/containers/${SLURM_JOBID}_$(basename ${CONT}).squashfs}"
: "${CONTAINER_PRELOAD_LUSTRE:=0}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
#: "${LOGDIR:=./results}"
: "${NSYSCMD:=""}"
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${NCCL_TEST:=0}"
: "${SYNTH_DATA:=0}"
: "${EPOCH_PROF:=0}"
: "${DISABLE_CG:=0}"
: "${WORK_DIR:=/workspace/bert}"
: "${UNITTEST:=0}"
readonly _cont_name=language_model
LOG_BASE="${DATESTAMP}"
SPREFIX="language_model_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}"

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

#if [ ${NVTX_FLAG--1} -gt 0 ] ||  [ ${TIME_TAGS--1} -gt 0 ]; then
#export MAX_STEPS=100
#fi

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
trap cleanup_preload_lustre EXIT

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

CONT_MOUNTS=$(func_get_container_mounts $(func_update_file_path_for_ci mounts.txt ${PWD}/language_model/pytorch))
CONT_MOUNTS="${CONT_MOUNTS},${LOGDIR}:/results"

if [[ "${NVTX_FLAG}" -gt 0 ]]; then
    CONT_MOUNTS="${CONT_MOUNTS},${NVMLPERF_NSIGHT_LOCATION}:/nsight"
fi
if [ "${API_LOGGING}" -eq 1 ]; then
    CONT_MOUNTS="${CONT_MOUNTS},${API_LOG_DIR}:/logs"
fi

# Setup directories
( umask 0002; mkdir -p "${LOGDIR}" )

# Setup container
echo MELLANOX_VISIBLE_DEVICES="${MELLANOX_VISIBLE_DEVICES:-}"
#srun --ntasks="$((SLURM_JOB_NUM_NODES))" --container-image="${CONT_FILE}" --container-name="${_cont_name}" true
#srun -N1 -n1 --container-name="${_cont_name}" ibv_devinfo --list
#srun -N1 -n1 --container-name="${_cont_name}" nvidia-smi topo -m

# Run NCCL test (700 MB FP16 allreduce)
#srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
#        --container-image=${CONT_FILE} \
#        all_reduce_perf_mpi -b 85M -e 680M -f 2 -d half

echo "NCCL_TEST = ${NCCL_TEST}"
if [[ ${NCCL_TEST} -eq 1 ]]; then
    (srun --mpi=pmix --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" \
         --container-name="${_cont_name}" all_reduce_perf_mpi -b 21M -e 672M -d half -G 1 -f 2 ) |& tee "${LOGDIR}/${SPREFIX}_nccl.log"

fi

# Run experiments
for _experiment_index in $(seq -w 1 "${NEXP}"); do
(
    echo "Beginning trial ${_experiment_index} of ${NEXP}"
	echo ":::DLPAL ${CONT} ${SLURM_JOB_ID} ${SLURM_JOB_NUM_NODES} ${SLURM_JOB_NODELIST}"

    # Clear caches
    if [ "${CLEAR_CACHES}" -eq 1 ]; then
        srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 bash -c "echo -n 'Clearing cache on ' && hostname && sync && sudo /sbin/sysctl vm.drop_caches=3"
        srun -N 1 --ntasks=1 --ntasks-per-node=1 apptainer exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert $CONT python -c "exec(\"from mlperf_logger import mllogger\nmllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)\")"
        srun --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} apptainer exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert $CONT python -c "exec(\"import torch\ntorch.cuda.empty_cache()\")"
    fi

    # Run experiment
	srun -l --ntasks=${SLURM_NTASKS}  --ntasks-per-node=${DGXNGPU} apptainer exec --nv -B "${CONT_MOUNTS}" -B $PWD:/workspace/bert $CONT bash /workspace/bert/run_and_time_multi.sh) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done



#      srun -l --ntasks="$(( SLURM_JOB_NUM_NODES * DGXNGPU ))" --ntasks-per-node="${DGXNGPU}" --container-name="${_cont_name}" --container-mounts="${CONT_MOUNTS}" --container-workdir=${WORK_DIR} "run_and_time.sh"
#    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"

#    # compliance checker
#    srun --ntasks=1 --nodes=1 --container-name="${_cont_name}" \
#         --container-mounts="$(realpath ${LOGDIR}):/results"   \
#         --container-workdir="/results"                        \
#         python3 -m mlperf_logging.compliance_checker --usage training \
#         --ruleset "${MLPERF_RULESET}"                                 \
#         --log_output "/results/compliance_${DATESTAMP}.out"           \
#         "/results/${LOG_BASE}_${_experiment_index}.log" \
#	|| true


# Cleanup: performed by cleanup_preload_lustre (see above) on EXIT trap
