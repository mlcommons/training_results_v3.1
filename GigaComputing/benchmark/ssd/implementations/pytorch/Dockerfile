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

ARG FROM_IMAGE_NAME=gitlab-master.nvidia.com/dl/dgx/pytorch:23.09-py3-stage

FROM ${FROM_IMAGE_NAME} AS base-image

################################################################################
# DALI box iou builder image
################################################################################
FROM base-image AS dali-box-iou-builder
WORKDIR /workspace/csrc
COPY csrc/dali_box_iou ./dali_box_iou

# Build CUDA box iou (written as a DALI extension)
RUN cd dali_box_iou && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
################################################################################


################################################################################
# DALI proposal matcher builder image
################################################################################
FROM base-image AS dali-proposal-matcher-builder
WORKDIR /workspace/csrc
COPY csrc/dali_proposal_matcher ./dali_proposal_matcher

# Build CUDA proposal matcher (written as a DALI extension)
RUN cd dali_proposal_matcher && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make
################################################################################


################################################################################
# RetinaNet
################################################################################
FROM base-image

# Build args
ARG MAX_JOBS=4
ARG TORCH_CUDNN_V8_API_ENABLED=1

# Set env vars
ENV MAX_JOBS=${MAX_JOBS}
ENV TORCH_CUDNN_V8_API_ENABLED=${TORCH_CUDNN_V8_API_ENABLED}

# Install dependencies for system configuration logger
RUN apt-get update && \
    apt-get install -y --no-install-recommends infiniband-diags pciutils && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace/ssd

# Copy code
COPY . .

# Remove the container's pycocotools in favor of the nvidia optimized version (installed from requirements.txt)
RUN pip uninstall -y pycocotools

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# JET
RUN pip install jet-api --upgrade --extra-index-url https://sc-hw-artf.nvidia.com/api/pypi/hw-joc-pypi/simple --extra-index-url https://sc-hw-artf.nvidia.com/api/pypi/compute-pypi/simple

# Copy DALI box iou and proposal matcher
COPY --from=dali-box-iou-builder /workspace/csrc/dali_box_iou/build/lib_box_iou.so /usr/local/lib/lib_box_iou.so
COPY --from=dali-proposal-matcher-builder /workspace/csrc/dali_proposal_matcher/build/lib_proposal_matcher.so /usr/local/lib/lib_proposal_matcher.so
################################################################################
