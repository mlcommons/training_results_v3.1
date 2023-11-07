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

import os
import random
import ctypes
from types import MethodType

import torch
import torch._dynamo
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from pytorch_lightning.plugins.io import TorchCheckpointIO
from pytorch_lightning import seed_everything

from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from megatron.core import parallel_state

import mlperf_logging_utils
from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger
from common import MultiprocessCheckpointIO

import debug_layer_stats

def l2_promote():
    _libcudart = ctypes.CDLL('libcudart.so')

    # Check what's the device limit for current device, should be 64 by default
    pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    result = _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))

    # Get the device limit again, should be 128
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    logging.info('L2 promotion: %d B', pValue[0])


@hydra_runner()
def main(cfg) -> None:
    mllogger.start(key=constants.INIT_START)

    if cfg.model.get("inductor", False):
        # Disable dynamic shape
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.automatic_dynamic_shapes = False

    # Promote L2 fetch to 128 bytes
    l2_promote()

    seed = random.SystemRandom().randint(0, 2**32 - 1)
    mllogger.event(key=constants.SEED, value=seed)
    seed_everything(seed)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = cfg.model.optim.get("name") == "distributed_fused_adam"

    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.model.capture_cudagraph_iters >= 0:
        # Required by CUDA graph with DDP
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

        # Hack to avoid CUDA graph issue with AMP, PyTorch Lightning doesn't support
        # changing autocast arguments for now.
        # https://github.com/pytorch/pytorch/blob/v1.13.1/torch/cuda/graphs.py#L234
        def amp_autocast_init(self, *args, **kwargs):
            if "cache_enabled" not in kwargs:
                kwargs["cache_enabled"] = False
            return self.__orig_init__(*args, **kwargs)

        torch.autocast.__orig_init__ = torch.autocast.__init__
        torch.autocast.__init__ = amp_autocast_init

    plugins = []

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.trainer.precision in [16, "bf16"]:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 65536.0),
                growth_interval=cfg.model.get("native_amp_growth_interval", 1000),
                hysteresis=cfg.model.get("hysteresis", 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device="cuda", scaler=scaler))

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())

    callbacks = []
    cb = mlperf_logging_utils.MLPerfLoggingCallback(logger=mllogger, train_log_interval=100,
                                                    global_batch_size=cfg.model.global_batch_size)
    cb.save_full_cfg(cfg)
    callbacks.append(cb)

    if debug_layer_stats.ENABLED:
        dg = debug_layer_stats.DebugCallback()
        callbacks.append(dg)

    checkpoint_io = MultiprocessCheckpointIO(
        checkpoint_io=TorchCheckpointIO(),
    )
    plugins.append(checkpoint_io)

    trainer = Trainer(
        plugins=plugins,
        strategy=strategy,
        callbacks=callbacks,
        enable_progress_bar=False,
        **cfg.trainer,
    )

    exp_manager(trainer, cfg.exp_manager)
    # update resume from checkpoint found by exp_manager
    if cfg.model.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # Re-order communicator
    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        self._orig_setup_distributed(global_rank, world_size)

        dummy = torch.randn(64, device="cuda", dtype=torch.float16)
        logging.info(f"Warmup allreduce with DDP communicator")
        for _ in range(20):
            torch.distributed.all_reduce(dummy, group=parallel_state.get_data_parallel_group())

    trainer.strategy._orig_setup_distributed = trainer.strategy.setup_distributed
    trainer.strategy.setup_distributed = MethodType(setup_distributed, trainer.strategy)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronLatentDiffusion(cfg.model, trainer)

    # Warmup the model with random data
    with torch.cuda.stream(torch.cuda.Stream()):
        n, c, h = cfg.model.micro_batch_size, cfg.model.channels, cfg.model.image_size
        x = torch.randn((n, c, h, h), dtype=torch.float32, device="cuda")
        t = torch.randint(77, (n,),  device="cuda")
        cc = torch.randn((n, 77, cfg.model.unet_config.context_dim), dtype=torch.float32, device="cuda")
        model = model.cuda()
        for _ in range(5):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model.model.model.diffusion_model(x, t, context=cc)
            grad = torch.randn_like(out)
            out.backward(grad)
            model.zero_grad()
    checkpoint_io.setup(model.state_dict())

    trainer.fit(model)

    # Since we created checkpoint in a new process, we wait to make sure the last checkpoint is saved
    checkpoint_io.teardown()

    trainer.strategy.barrier()


if __name__ == "__main__":
    main()
