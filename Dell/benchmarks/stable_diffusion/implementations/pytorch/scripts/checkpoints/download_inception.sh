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

: "${OUTPUT_DIR:=/checkpoints/inception}"

while [ "$1" != "" ]; do
    case $1 in
        -o | --output-dir )       shift
                                  OUTPUT_DIR=$1
                                  ;;
    esac
    shift
done

FID_WEIGHTS_URL='https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
FID_WEIGHTS_SHA1="bd836944fd6db519dfd8d924aa457f5b3c8357ff"

wget -N -P ${OUTPUT_DIR} ${FID_WEIGHTS_URL}
echo "${FID_WEIGHTS_SHA1}  ${OUTPUT_DIR}/pt_inception-2015-12-05-6726825d.pth"                    | sha1sum -c
