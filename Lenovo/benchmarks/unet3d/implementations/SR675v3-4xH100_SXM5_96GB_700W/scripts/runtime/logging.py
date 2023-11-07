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

import dllogger as logger
from dllogger import StdOutBackend, Verbosity, JSONStreamBackend
from mlperf_common.logging import MLLoggerWrapper
from mlperf_common.scaleoutbridge import init_bridge
from mlperf_common.frameworks.mxnet import MXNetProfilerHandler, MPICommunicationHandler

mllogger = MLLoggerWrapper(MPICommunicationHandler(), value=None)
sbridge = init_bridge(MXNetProfilerHandler(), MPICommunicationHandler(), mllogger)


def get_logger(params, eval_ranks, global_rank):
    backends = []
    if global_rank == eval_ranks[0]:
        if params.verbose:
            backends += [StdOutBackend(Verbosity.VERBOSE)]
        if params.log_dir:
            backends += [JSONStreamBackend(Verbosity.VERBOSE, params.log_dir)]
    logger.init(backends=backends)
    return logger


def mlperf_run_param_log(flags):
    mllogger.event(key=mllogger.constants.OPT_NAME, value=flags.optimizer)
    mllogger.event(key=mllogger.constants.OPT_BASE_LR, value=flags.learning_rate)
    mllogger.event(key=mllogger.constants.OPT_LR_WARMUP_EPOCHS, value=flags.lr_warmup_epochs)
    # mllogger.event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=flags.lr_warmup_factor)
    mllogger.event(key=mllogger.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS, value=flags.lr_decay_epochs)
    mllogger.event(key=mllogger.constants.OPT_LR_DECAY_FACTOR, value=flags.lr_decay_factor)
    mllogger.event(key=mllogger.constants.OPT_WEIGHT_DECAY, value=flags.weight_decay)
    mllogger.event(key="opt_momentum", value=flags.momentum)
    mllogger.event(key="oversampling", value=flags.oversampling)
    mllogger.event(key="training_input_shape", value=flags.input_shape)
    mllogger.event(key="validation_input_shape", value=flags.val_input_shape)
    mllogger.event(key="validation_overlap", value=flags.overlap)
