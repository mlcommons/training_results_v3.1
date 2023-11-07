# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.04-py3
FROM ${FROM_IMAGE_NAME}

# Install dependencies for system configuration logger
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        infiniband-diags \
        pciutils \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libibumad3 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /workspace/object_detection

ENV CUDA_MODULE_LOADING=EAGER
ENV TORCH_CUDNN_V8_API_ENABLED 1

# Control number of workers for ninja, so to avoid OOM when building on
# machines with less than 16GB.  Peak memory usage per job is ~1.8GB
ENV MAX_JOBS 6

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy detectron code and build
# Issue: using COPY or -e may cause using the local build cache, find a way to pip
# install without cache -e and --no-cache-dir
COPY . .
RUN pip install -e maskrcnn

ENV OMP_NUM_THREADS=1
ENV OPENCV_FOR_THREADS_NUM=1

