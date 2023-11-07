#!/bin/bash
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

set -x

TMP_SQSH="${SCRATCH_SPACE}/tmp.sqsh"
REQ_SIZE="573332353024"

# Check if squash file exists and is the right size
if [[ -f ${TMP_SQSH} ]]; then
  TMP_SQSH_SIZE=$(stat --printf='%s' ${TMP_SQSH})

  if [[ ${TMP_SQSH_SIZE} -eq ${REQ_SIZE} ]]; then
    echo "squash file exists and is not corrupted"
    exit 0
  else
    echo "squash file exists but is corrupted, copying..."
  fi
else
  echo "squash file doesn't exist, copying..."
fi

dd bs=4M if=${LOCALDISK_FROM_SQUASHFS} of=${SCRATCH_SPACE}/tmp.sqsh oflag=direct
