# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

ARG SM="80;90"
ARG ENABLE_MULTINODES=ON

ARG HWLOC_VERSION=2.4.1
ARG RELEASE=true

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        clang-format \
	libboost-serialization-dev \
        libtbb-dev \
        libaio-dev \
        #   Required to build RocksDB.
        libgflags-dev \
        zlib1g-dev libbz2-dev libsnappy-dev liblz4-dev libzstd-dev \
        #   Required to build RdKafka.
        zlib1g-dev libzstd-dev \
        libssl-dev libsasl2-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/usr/local/bin:$PATH

# CUDA-Aware hwloc
RUN cd /opt/hpcx/ompi/include/openmpi/opal/mca/hwloc/hwloc201 && rm -rfv hwloc201.h hwloc/include/hwloc.h
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp https://download.open-mpi.org/release/hwloc/v2.4/hwloc-${HWLOC_VERSION}.tar.gz && \
    mkdir -p /var/tmp && tar -x -f /var/tmp/hwloc-${HWLOC_VERSION}.tar.gz -C /var/tmp && \
    cd /var/tmp/hwloc-${HWLOC_VERSION} && \
    ./configure CPPFLAGS="-I/usr/local/cuda/include/ -L/usr/local/cuda/lib64/" LDFLAGS="-L/usr/local/cuda/lib64" --enable-cuda && \
    make -j$(nproc) && make install && \
    rm -rf /var/tmp/hwloc-${HWLOC_VERSION} /var/tmp/hwloc-${HWLOC_VERSION}.tar.gz

ENV CPATH=/usr/local/include:$CPATH

# Env variables for NCCL
ENV NCCL_LAUNCH_MODE=PARALLEL

# ENV variables for Sharp
ENV SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
    SHARP_COLL_LOCK_ON_COMM_INIT=1 \
    SHARP_COLL_LOG_LEVEL=3 \
    HCOLL_ENABLE_MCAST=0

RUN ln -s /usr/lib/x86_64-linux-gnu/libibverbs.so.1.14.39.0 /usr/lib/x86_64-linux-gnu/libibverbs.so

WORKDIR /workspace/dlrm
COPY . .

RUN pip3 install --no-cache-dir --no-build-isolation -r requirements.txt

# FIXME: Temporal workaround to make HugeCTR compile correctly
#  for the latest PyT base image (following conda removal):
RUN pip3 uninstall -y tbb

ENV PYTHON_VERSION=3.10
# HugeCTR
RUN if [ "$RELEASE" = "true" ]; \
    then \
	cd /workspace/dlrm/ && \
	git clone --recurse-submodules -b v23.08.00 https://github.com/NVIDIA-Merlin/HugeCTR.git hugectr && \
    	cd hugectr && \
    	git log -n 1 && \
    	mkdir build && cd build && \
    	cmake -DCMAKE_BUILD_TYPE=Release -DSM=$SM -DUSE_CUDART_STATIC=ON \
    	    -DENABLE_MULTINODES=$ENABLE_MULTINODES -DSHARP_A2A=OFF .. && \
    	make -j$(nproc) && make install && \
    	chmod +x /usr/local/hugectr/bin/* && \
    	chmod +x /usr/local/hugectr/lib/* && \
    	rm -rf /workspace/dlrm/hugectr && \
	rm /usr/lib/x86_64-linux-gnu/libibverbs.so; \
    else \
      echo "Build container for development successfully!"; \
    fi

ENV PATH=/usr/local/hugectr/bin:$PATH \
    LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
