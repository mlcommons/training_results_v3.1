# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
import torch
import numpy as np
from mlperf_logging import mllog


from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)


def barrier():
    """
    Works as a temporary distributed barrier, currently pytorch
    doesn't implement barrier for NCCL backend.
    Calls all_reduce on dummy tensor and synchronizes with GPU.
    """
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


def get_rank():
    """
    Gets distributed rank or returns zero if distributed is not initialized.
    """
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def generate_seeds(rng, size):
    seeds = [rng.randint(0, 2**32 - 1) for _ in range(size)]
    return seeds

def broadcast_seeds(seeds, device):
    if torch.distributed.is_initialized():
        seeds_tensor = torch.LongTensor(seeds).to(device)
        torch.distributed.broadcast(seeds_tensor, 0)
        seeds = seeds_tensor.tolist()
    return seeds

def set_seeds(args):
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda')

    # make sure that all workers has the same master seed
    args.seed = broadcast_seeds(args.seed, device)

    local_seed = (args.seed + get_rank()) % 2**32
    print(get_rank(), "Using seed = {}".format(local_seed))
    torch.manual_seed(local_seed)
    np.random.seed(seed=local_seed)
    return local_seed
