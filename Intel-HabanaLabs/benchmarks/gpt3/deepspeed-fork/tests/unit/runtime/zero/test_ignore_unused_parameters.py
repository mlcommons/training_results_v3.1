# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import torch
from unit.common import DistributedTest
from unit.simple_model import UnusedParametersModel, random_dataloader
from deepspeed.ops.op_builder import CPUAdamBuilder
from unit.hpu import *

import deepspeed


@pytest.mark.parametrize('ignore_unused_parameters', [False, True])
class TestStage2IgnoreUnusedParameters(DistributedTest):
    world_size = 1

    def test(self, ignore_unused_parameters):
        use_cpu_offload = True

        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 2,
            "steps_per_print": 1,
            "zero_optimization": {
                "stage": 2,
                "cpu_offload": use_cpu_offload,
                "ignore_unused_parameters": ignore_unused_parameters
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-3
                }
            },
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            }
        }
        hidden_dim = 4
        dtype = torch.half
        if bool(pytest.use_hpu) == True:
            if os.getenv("REPLACE_FP16", default=None):
                config_dict["fp16"]["enabled"] = False
                config_dict["fp32"] = {"enabled": True}
                dtype = torch.float
                if get_hpu_dev_version() == "Gaudi":
                    config_dict["communication_data_type"] = 'bfp16'
            hpu_flag, msg = is_hpu_supported(config_dict)
            if not hpu_flag:
                pytest.skip(msg)

        model = UnusedParametersModel(hidden_dim=hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=dtype)

        def _loop():
            for n, batch in enumerate(data_loader):
                loss = model(batch[0], batch[1])
                model.backward(loss)
                model.step()

        if ignore_unused_parameters:
            _loop()
        else:
            with pytest.raises(AssertionError) as e:
                _loop()
            assert e.value.args and 'ignore_unused_parameters' in e.value.args[0]
