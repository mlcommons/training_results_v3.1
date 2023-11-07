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

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=3.0.0}"
: "${DATADIR:=/raid/datasets/train-val-recordio-passthrough}"
: "${LOGDIR:=$(pwd)/results}"
: "${COPY_DATASET:=}"

echo $COPY_DATASET

if [ ! -z $COPY_DATASET ]; then
    readonly copy_datadir=$COPY_DATASET
    mkdir -p "${DATADIR}"
    ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
    ls ${DATADIR}
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=image_classification
_cont_mounts=("--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results")

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME} / ${DGX_PRETTY_NAME:-???} ${DGX_OTA_VERSION:-${DGX_SWBUILD_VERSION:-???}}"
)
export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS SEED)
_config_env+=(MLPERF_SUBMISSION_ORG)
_config_env+=(MLPERF_SUBMISSION_PLATFORM)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
nvidia-docker run --rm --init --detach \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    --name="${_cont_name}" "${_cont_mounts[@]}" \
    "${CONT}" sleep infinity
#make sure container has time to finish initialization
sleep 30
docker exec -it "${_cont_name}" true

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        docker exec -it "${_cont_name}" python -c "
from mlperf_log_utils import mllogger
mllogger.mlperf_submission_log(mllogger.constants.RESNET)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo -S /sbin/sysctl vm.drop_caches=3 < password.txt
            docker exec -it "${_cont_name}" python -c "
from mlperf_log_utils import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        docker exec -it "${_config_env[@]}" "${_cont_name}" \
	       mpirun --allow-run-as-root --bind-to none --np ${DGXNGPU} ./run_and_time.sh
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
