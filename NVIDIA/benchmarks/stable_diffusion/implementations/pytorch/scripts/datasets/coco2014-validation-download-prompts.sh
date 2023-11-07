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

: "${OUTPUT_DIR:=/datasets/coco2014}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )     shift
                                OUTPUT_DIR=$1
                                ;;
    esac
    shift
done

mkdir -p ${OUTPUT_DIR}
wget -O ${OUTPUT_DIR}/val2014_30k.tsv -c "https://cloud.mlcommons.org/index.php/s/training_stable_diffusion/download?path=/datasets/coco2014&files=val2014_30k.tsv"
