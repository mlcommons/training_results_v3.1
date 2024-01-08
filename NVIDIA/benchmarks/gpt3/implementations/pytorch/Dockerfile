# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.09-py3
FROM ${FROM_IMAGE_NAME}

# Document build setup
ARG FROM_IMAGE_NAME
ENV CUSTOM_FROM_IMAGE_NAME ${FROM_IMAGE_NAME}

# Custom libraries version
WORKDIR /workspace/

## 0. Fix for https://github.com/NVIDIA/nccl/issues/957
ARG NCCL_COMMIT_ID=4365458757e4107ecbf629b2fd6e0e19a5d237c2

RUN export DEBIAN_FRONTEND=noninteractive &&        \
    apt-get update &&                               \
    apt-get remove -y libnccl2 &&                   \
    git clone https://github.com/NVIDIA/nccl.git && \
    cd nccl &&                                      \
    echo "NCCL_COMMIT_ID=${NCCL_COMMIT_ID}" &&      \
    git checkout "${NCCL_COMMIT_ID}" &&             \
    make -j24 install
ENV NCCL_VERSION "${NCCL_VERSION}-fix-${NCCL_COMMIT_ID}"

## 1. Apex
ARG APEX_REVISION=SKIP
ENV CUSTOM_APEX_REVISION ${APEX_REVISION}
ARG APEX_MAX_JOBS=4

RUN if [ "${APEX_REVISION}" != SKIP ]; then \
      git clone https://github.com/NVIDIA/apex && \
      cd apex && \
      echo APEX_REVISION=${APEX_REVISION} && \
      git checkout ${APEX_REVISION} && \
      MAX_JOBS=${APEX_MAX_JOBS} pip3 install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_layer_norm" --global-option="--distributed_adam" --global-option="--deprecated_fused_adam" ./ \
    ; fi


## 2. Transformer Engine
# the following two variables are needed when building TE
ENV NVTE_WITH_USERBUFFERS=1
ENV MPI_HOME=/usr/local/mpi
ARG TE_REVISION=09ba84278a0e5b2eff80239599fbfc5e6beb389f
ENV CUSTOM_TE_REVISION ${TE_REVISION}

RUN if [ "${TE_REVISION}" != SKIP ]; then \
      pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi


## 3. NeMo
ARG NEMO_REVISION=bd0b818ac2eb420b1cea19eb8bcc6215c87b06d0
ENV CUSTOM_NEMO_REVISION ${NEMO_REVISION}
ARG NEMO_BASE_VERSION=r1.20.0
ENV CUSTOM_NEMO_BASE_VERSION ${NEMO_BASE_VERSION}

### Base version
RUN if [ "${NEMO_REVISION}" == SKIP ]; then \
      if [ -d /opt/bignlp/NeMo ]; then \
        ln -s /opt/bignlp/NeMo \
      ; else \
        echo "Error: NEMO_REVISION=SKIP but there is no BigNLP NeMo installation in base image." && \
        exit 1 \
      ; fi \
    ; else \
      git clone https://github.com/NVIDIA/NeMo.git && \
      cd NeMo && \
      echo NEMO_REVISION=${NEMO_REVISION} && \
      git checkout ${NEMO_REVISION} && \
      pip uninstall -y nemo-toolkit && \
      pip install "cython<3.0.0" && \
      pip install --no-build-isolation -e ".[nlp]" \
    ; fi

# Install flash-attention 2.0.8
RUN MAX_JOBS=4 pip install --upgrade --no-deps flash-attn==2.0.8

### Make (has to be called after all changes to repo)
RUN cd NeMo && \
      cd nemo/collections/nlp/data/language_modeling/megatron && \
      make

## 4. Megatron-core
ARG MEGATRON_REVISION=df7271285d3e29c13865515a51dbbd6e25f68f5f
ENV CUSTOM_MEGATRON_REVISION ${MEGATRON_REVISION}

RUN if [ "${MEGATRON_REVISION}" != SKIP ]; then \
      pip uninstall -y megatron-core && \
      pip install git+https://github.com/nvidia/Megatron-LM.git@${MEGATRON_REVISION} \
    ; fi


## 4.5 Add old Nsight
#RUN installNSYS.sh 2023.2.1

## 5. Benchmark dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Benchmark code
WORKDIR /workspace/llm

COPY . .
ENV PYTHONPATH "/workspace/llm:/workspace/NeMo:${PYTHONPATH}"

