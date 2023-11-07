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

: "${OUTPUT_DIR:=/checkpoints/clip}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

CLIP_WEIGHTS_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin"
CLIP_WEIGHTS_SHA256="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"

CLIP_CONFIG_URL="https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/raw/main/open_clip_config.json"

wget -N -P ${OUTPUT_DIR} ${CLIP_WEIGHTS_URL}
wget -N -P ${OUTPUT_DIR} ${CLIP_CONFIG_URL}
echo "${CLIP_WEIGHTS_SHA256}  ${OUTPUT_DIR}/open_clip_pytorch_model.bin"                    | sha256sum -c
