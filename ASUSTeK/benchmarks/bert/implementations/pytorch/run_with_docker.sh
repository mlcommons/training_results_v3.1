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

set -euxo pipefail

#######################################
# Print mount locations from file.
# Arguments:
#   arg1 - path to file containing volume mounting points
# Returns:
#   String containing comma-separated mount pairs list
#######################################
func_get_container_mounts() {
  echo $(envsubst <<< $(sed '/^$/d' ${1} | sed '/^#/d' | sed 's/^[ ]*\(.*\)[ ]*/--volume=\1 /' | tr '\n' ' '))
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


# Vars with defaults
: "${NEXP:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=3.1.0}"
: "${LOGDIR:=$(pwd)/results}"
: "${MLPERF_MODEL_CONSTANT:=constants.BERT}"

: "${CONFIG_FILE:="./config_${DGXSYSTEM}.sh"}"
: "${_logfile_base:="${LOGDIR}/${DATESTAMP}"}"
: "${DGXNGPU:=1}"
: "${NV_GPU:="${CUDA_VISIBLE_DEVICES}"}"

readonly docker_image=${CONT:-"nvcr.io/SET_THIS_TO_CORRECT_CONTAINER_TAG"}
readonly _cont_name=language_model
_cont_mounts=("--volume=${DATADIR_PHASE2}:/workspace/data_phase2" "--volume=${CHECKPOINTDIR_PHASE1}:/workspace/phase1 ")
_cont_mounts+=("--volume=${EVALDIR}:/workspace/evaldata" "--volume=${LOGDIR}:/results")

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${CONFIG_FILE} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach --gpus='"'device=${NV_GPU}'"' \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" ${_cont_mounts[@]} \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true

readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        docker exec -it "${_config_env[@]}" "${_cont_name}" \
               ${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"

      if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec -it "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || true
      fi
    sleep 30
done
