# Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.09-py3
FROM ${FROM_IMAGE_NAME}

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y sox

RUN apt-get install -y --no-install-recommends numactl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/rnnt

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt

COPY . .
