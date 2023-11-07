#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

: "${OUTPUT_DIR:=/checkpoints/sd}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

SD_WEIGHTS_URL='https://huggingface.co/stabilityai/stable-diffusion-2-base/resolve/main/512-base-ema.ckpt'
SD_WEIGHTS_SHA256="d635794c1fedfdfa261e065370bea59c651fc9bfa65dc6d67ad29e11869a1824"

wget -N -P ${OUTPUT_DIR} ${SD_WEIGHTS_URL}
echo "${SD_WEIGHTS_SHA256}  ${OUTPUT_DIR}/512-base-ema.ckpt"                    | sha256sum -c
