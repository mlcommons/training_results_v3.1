# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose, get_dtypes
import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None


def run_vector_add_reference(a, b, gamma):
    return torch_fallback_kernels.vector_add_fallback(a, b, gamma)


def run_vector_add_ds(a, b, gamma):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return inference_module._vector_add(a, b, gamma)


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 128, 255])
@pytest.mark.parametrize("channels", [512, 1232, 4096])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_vector_add(batch, sequence, channels, dtype):
    a_ds = torch.randn((batch, sequence, channels), dtype=dtype, device=get_accelerator().device_name())
    b_ds = torch.randn((channels), dtype=dtype, device=get_accelerator().device_name())
    import random
    gamma = random.random()

    a_ref = a_ds.clone().detach()
    b_ref = b_ds.clone().detach()

    ds_out = run_vector_add_ds(a_ds, b_ds, gamma)
    ref_out = run_vector_add_reference(a_ref, b_ref, gamma)
    if not allclose(ds_out, ref_out):
        print((ds_out - ref_out).abs().max())
        assert (allclose(ds_out, ref_out))
