# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
from bisect import bisect_right

import torch


class WarmupMultiStepLR:
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
        scale_window=100,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear", "mlperf_linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        if self.warmup_method == "constant":
            self.warmup_method_index = 1
        elif self.warmup_method == "linear":
            self.warmup_method_index = 2
        elif self.warmup_method == "mlperf_linear":
            self.warmup_method_index = 3
        else:
            self.warmup_method_index = 0
        self.dynamic_loss_scale = True
        self.scale_window = scale_window
        self.step_properties = torch.cuda.FloatTensor([0,0,65536,0,0]) # 65536 is initial value of loss scaler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        for group, lr in zip(optimizer.param_groups, self.get_lr()):
            group['lr'] = lr

    def step(self, overflow_buf):
        import maskrcnn_benchmark.Syncfree as sf
        sf.step_scheduler_loss_scaler_cuda(
                overflow_buf,
                self.step_properties,
                self.warmup_factor,
                self.warmup_method_index,
                self.warmup_iters,
                self.gamma,
                self.milestones[0],
                self.milestones[1],
                self.base_lrs[0],
                self.base_lrs[1],
                1 if self.dynamic_loss_scale else 0,
                2.0,
                self.scale_window)

    def get_lr(self):
        return [self.step_properties[0], self.step_properties[1]]
