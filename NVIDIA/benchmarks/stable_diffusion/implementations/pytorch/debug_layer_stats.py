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
from typing import Any

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.types import STEP_OUTPUT
except:
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.types import STEP_OUTPUT

ENABLED = os.environ.get("DEBUG_LAYER_STATS", "0") == "1"


def stat_report(pl_module):
    print(f"LAYER STATISTICS DEBUG REPORT")
    if ENABLED:
        print(f"NAME, SHAPE, GRAD, MEAN, MEAN_ABS, STD, MIN, MAX")
        for name, param in pl_module.named_parameters():
            mean = param.data.mean()
            mean_abs = param.data.abs().mean()
            std = param.data.std()
            req_grad = param.requires_grad
            min_val = param.data.min()
            max_val = param.data.max()
            print(
                f"{name}; {list(param.shape)}; {req_grad}; {mean:9.6f}; {mean_abs:9.6f}; {std:9.6f}; {min_val:9.6f}; {max_val:9.6f}"
            )
    else:
        print(f"DebugCallback is disabled. Set DEBUG_CALLBACK=1 to enable.")


class DebugCallback(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()
        pass

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Print weights of all the parameters (mean, mean(abs), std)
        stat_report(pl_module)

    def on_fit_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        stat_report(pl_module)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                             batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                           outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        pass
