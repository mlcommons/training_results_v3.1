# Copyright (c) 2020-2023, Google CORPORATION. All rights reserved.
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
set -eox pipefail

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"

# Vars without defaults
CLUSTER_NAME=mlperf-cluster
TPU_TOPOLOGY=16x16
EXP_NAME=C4SpmdGpt3AdamDataParallel16x16x16Int8
NUM_SLICES=16
ACCELERATOR=tpu-v5-lite-podslice

# Vars with defaults
TAG=${JAX_LIBTPU_IMAGE}
NUM_CHIPS=$(( $(echo "${TPU_TOPOLOGY}" |  sed 's/x/*/g') ))
NUM_CHIPS_PER_NODE=4
NUM_NODES=$(( $NUM_CHIPS / $NUM_CHIPS_PER_NODE ))

TIMESTAMP=$(date "+%Y%m%d-%H%M%S")
ACCELERATOR_TYPE=$(echo "${ACCELERATOR}" | awk -F"-" '{ print $2 }')
IMAGE=${TAG}
ENABLE_LOCAL_AQT=true

GS_PREFIX=mlperf-exp/submissions/${USER}/${EXP_NAME}/${NUM_SLICES}_${ACCELERATOR_TYPE}_${NUM_CHIPS}
CLIENT_YAML_FILE="${SCRIPTS_DIR}/jobset_${NUM_SLICES}_${ACCELERATOR_TYPE}_${NUM_CHIPS}.yaml"
# replace placeholder in template CLIENT_YAML_FILE
patterns=(
   "<USER>" "${USER}"
   "<NUM_SLICES>" "${NUM_SLICES}"
   "<NUM_CHIPS>" "${NUM_CHIPS}"
   "<NUM_NODES>" "${NUM_NODES}"
   "<TPU_TOPOLOGY>" "${TPU_TOPOLOGY}"
   "<PAX_DATE>" \""${PAX_DATE}"\"  # int type not allowed in k8s env var convert to string
   "<EXP_NAME>" "${EXP_NAME}"
   "<GS_PREFIX>" "${GS_PREFIX}"
   "<NUM_CHIPS_PER_NODE>" "${NUM_CHIPS_PER_NODE}"
   "<ACCELERATOR>" "${ACCELERATOR}"
   "<ACCELERATOR_TYPE>" "${ACCELERATOR_TYPE}"
   "<TIMESTAMP>" "${TIMESTAMP}"
   "<IMAGE>" "${IMAGE}"
   "<ENABLE_LOCAL_AQT>" \""${ENABLE_LOCAL_AQT}"\"  # bool type not allowed in k8s env var convert to string
)

sed_arg=$(
  for ((i = 0; i < ${#patterns[@]}; i += 2)); do
      # use "|" since "/" is in GS_PREFIX
      echo -n "s|${patterns[i]}|${patterns[i+1]}|g;"
  done
)
cat "${SCRIPTS_DIR}"/scripts/jobset_template.yaml | sed "$sed_arg" > "${CLIENT_YAML_FILE}"

GKE_JOB_ID=${USER}-mlperf-gpt3-benchmark-${ACCELERATOR_TYPE}-${NUM_CHIPS}-${TIMESTAMP}
echo "Run vars: id $GKE_JOB_ID  $TPU_TOPOLOGY num_slices $NUM_SLICES"
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

kubectl apply -f "${CLIENT_YAML_FILE}"

kubectl wait --for=condition=complete --timeout=120m job/"${GKE_JOB_ID}"-job-0

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="large language model"
echo "RESULT,${RESULT_NAME},${RESULT},${USER},${START_FMT}"