# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import os
import subprocess
import numpy as np
from mlperf_logging.mllog import constants
from mlperf_logging import mllog

from mlperf_common.frameworks.mxnet import MPICommunicationHandler
from mlperf_common.logging import MLLoggerWrapper

class MPIWrapper(MPICommunicationHandler):
    def __init__(self):
        super().__init__()
        from mpi4py import MPI
        self.MPI = MPI


    def allreduce(self, x):
        val = np.array(x, dtype=np.int32)
        result = np.zeros_like(val, dtype=np.int32)
        self._get_comm().Allreduce([val, self.MPI.INT], [result, self.MPI.INT])
        return result

    def rank(self):
        c = self.get_comm()
        return c.Get_rank()

mpiwrapper = MPIWrapper()
mllogger = MLLoggerWrapper(mpiwrapper, value=None)

def resnet_max_pool_log(input_shape, stride):
    downsample = 2 if stride == 2 else 1
    output_shape = (input_shape[0], 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_begin_block_log(input_shape):
    return input_shape


def resnet_end_block_log(input_shape):
    return input_shape


def resnet_projection_log(input_shape, output_shape):
    return output_shape


def resnet_conv2d_log(input_shape, stride, out_channels, bias):
    downsample = 2 if (stride == 2 or stride == (2, 2)) else 1
    output_shape = (out_channels, 
                    int(input_shape[1]/downsample), 
                    int(input_shape[2]/downsample))

    return output_shape


def resnet_relu_log(input_shape):
    return input_shape


def resnet_dense_log(input_shape, out_features):
    shape = (out_features)
    return shape


def resnet_batchnorm_log(shape, momentum, eps, center=True, scale=True, training=True):
    return shape
