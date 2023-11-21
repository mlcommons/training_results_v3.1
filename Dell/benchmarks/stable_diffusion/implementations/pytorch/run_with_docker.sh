#!/bin/bash

# Copyright © 2023 Dell Inc. or its subsidiaries.  All Rights Reserved.  Dell Technologies, 
# Dell and other trademarks are trademarks of Dell Inc.  or its subsidiaries. 
# Other trademarks may be trademarks of their respective owners."
#
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
NEXP=10
: "${NEXP:=10}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${CHECK_COMPLIANCE:=1}"
: "${MLPERF_RULESET:=3.1.0}"
#: "${DATADIR:=/raid/datasets/kits19}"
: "${LOGDIR:=$(pwd)/results}"
: "${SET_MAXQ_CLK:=0}"
: "${SET_MINEDP_CLK:=0}"
export RANDOM_SEED=$RANDOM
echo $RANDOM_SEED



# Other vars
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
#readonly _cont_name=image_segmentation
readonly _cont_name=stable_diffusion
_cont_mounts=("--volume=${DATADIR}:/datasets" "--volume=${LOGDIR}:/results" "--volume=${CHECKPOINTS}:/checkpoints" "--volume=${NEMOLOGS}:/nemologs" "--volume=${PWD}:/workspace/sd" "--volume=${NEMOLOGS}/hf_home:/hf_home")
# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
_config_env+=(MLPERF_HOST_OS)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env=$v"; done)

# Cleanup container
cleanup_docker() {
    docker container rm -f "${_cont_name}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Setup container
#docker run --gpus all --rm --init --detach \
#docker run --gpus all --rm --shm-size=20g --init --detach -w /workspace/sd \
docker run --gpus all --rm --init --detach -w /workspace/sd \
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

        fi
	
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            docker exec -it "${_cont_name}" python -c "
from mlperf_logging import mllog
mllogger = mllog.get_mllogger()
mllogger.event(key=mllog.constants.CACHE_CLEAR, value=True)"
	fi


        # Run experiment
        docker exec -it "${_config_env[@]}" "${_cont_name}" \
	       mpirun --allow-run-as-root --bind-to none -np ${DGXNGPU} ./run_and_time.docker.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"

        #rm -rf ${NEMOLOGS}/${EXP_NAME}/inference
	rm -rf ${NEMOLOGS}/stable-diffusion2-train*

      if [ "${CHECK_COMPLIANCE}" -eq 1 ]; then
      docker exec -it -e MLPERF_SUBMISSION_ORG -e MLPERF_SUBMISSION_PLATFORM "${_config_env[@]}" "${_cont_name}"  \
           python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "${MLPERF_RULESET}"                                 \
           --log_output "/results/compliance_${DATESTAMP}.out"           \
           "/results/${DATESTAMP}_${_experiment_index}.log" \
      || true
      fi
    sleep 30
done
