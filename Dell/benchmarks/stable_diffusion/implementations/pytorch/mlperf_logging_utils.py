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
import time
import torch
from typing import Any, Dict, Optional, Type

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.types import STEP_OUTPUT
except:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT

from mlperf_common.logging import MLLoggerWrapper, constants
from mlperf_common.frameworks.pyt import PyTCommunicationHandler

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)


def extract_step_from_ckpt_name(ckpt_name):
    ckpt_name = ckpt_name[ckpt_name.find("-step=") + len("-step=") :]
    ckpt_name = ckpt_name[: ckpt_name.find("-")]
    return int(ckpt_name)


def extract_timestamp_from_ckpt_name(ckpt_name):
    ckpt_name = ckpt_name[ckpt_name.find("-timestamp=") + len("-timestamp=") :]
    ckpt_name = ckpt_name[: ckpt_name.find("-")]
    return int(float(ckpt_name))


class MLPerfLoggingCallback(pl.callbacks.Callback):
    def __init__(self, logger, global_batch_size, train_log_interval=5, validation_log_interval=1):
        super().__init__()
        self.logger = mllogger
        self.train_log_interval = train_log_interval
        self.global_batch_size = global_batch_size
        self.validation_log_interval = validation_log_interval

        self.train_batch_start_time = time.perf_counter()
        self.train_batch_start_step = 0

        self.cfg = None

    def save_full_cfg(self, cfg):
        self.cfg = cfg

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Callback init is before DDP init, put them here to avoid wrong
        # device placement
        self.summed_loss = torch.zeros(1, device="cuda")
        self.summed_loss_n = 0

        mllogger.event(key=constants.GRADIENT_ACCUMULATION_STEPS, value=trainer.accumulate_grad_batches)
        mllogger.event(key=constants.GLOBAL_BATCH_SIZE, value=self.cfg.model.global_batch_size)

        mllogger.event(key=constants.OPT_NAME, value=constants.ADAMW)
        mllogger.event(key=constants.OPT_ADAMW_BETA_1, value=0.9)
        mllogger.event(key=constants.OPT_ADAMW_BETA_2, value=0.999)
        mllogger.event(key=constants.OPT_ADAMW_EPSILON, value=1e-08)
        mllogger.event(key=constants.OPT_ADAMW_WEIGHT_DECAY, value=0.01)

        mllogger.event(key=constants.OPT_BASE_LR, value=self.cfg.model.optim.lr)
        mllogger.event(key=constants.OPT_LR_WARMUP_STEPS, value=self.cfg.model.optim.sched.warmup_steps)
        
        mllogger.event(key=constants.TRAIN_SAMPLES, value=6513144)
        mllogger.event(key=constants.EVAL_SAMPLES, value=30000)

        mllogger.mlperf_submission_log(
            benchmark=constants.STABLE_DIFFUSION,
            num_nodes=self.cfg.trainer.num_nodes,
        )
        mllogger.log_init_stop_run_start()

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # No RUN_STOP here because it is after CLIP metric calculation
        # self.logger.end(constants.RUN_STOP, metadata=dict(status=constants.SUCCESS))
        pass

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if trainer.global_step % self.train_log_interval == 0:
            self.logger.start(
                key=constants.BLOCK_START, value="training_step", metadata={constants.STEP_NUM: trainer.global_step}
            )
            self.train_batch_start_time = time.perf_counter()
            self.train_batch_start_step = trainer.global_step

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        logs = trainer.callback_metrics
        self.summed_loss += logs["train/loss"]
        self.summed_loss_n += 1

        if (trainer.global_step - self.train_batch_start_step) == self.train_log_interval:
            self.logger.end(
                key=constants.BLOCK_STOP, value="training_step", metadata={constants.STEP_NUM: trainer.global_step}
            )

            throughput = (
                self.global_batch_size * self.train_log_interval / (time.perf_counter() - self.train_batch_start_time)
            )
            self.logger.event(
                key="tracked_stats",
                metadata={constants.STEP_NUM: trainer.global_step},
                value={
                    "throughput": throughput,
                    "loss": self.summed_loss.item() / (self.summed_loss_n + 1e-6),
                    "lr": logs["lr"].item(),
                },
            )
            self.summed_loss.fill_(0)
            self.summed_loss_n = 0
