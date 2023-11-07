# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright Habana Labs, Ltd. an Intel Company'''

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.ops.op_builder.torch_fallback_builder import TorchInferenceOpBuilder
from .inference_test_utils import allclose, get_dtypes
import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytestmark = pytest.mark.skip(reason="Inference ops are not available on this system")


def get_inference_modules():
    return [InferenceBuilder().load(), TorchInferenceOpBuilder().load()]


def run_transform4d_0213_reference(input, seq):
    return torch_fallback_kernels.transform4d_0213(input, seq)


def run_transform4d_0213_ds(inference_module, input, seq):
    batch_size = input.shape[0]
    heads = input.shape[1]
    head_dim = input.shape[2] // seq
    hidden_dim = heads * head_dim

    allocate_workspace_func = getattr(inference_module,
                                      f"allocate_workspace_{torch_fallback_kernels.dtype_names_dict[input.dtype]}")
    kernel_func = getattr(inference_module, f"transform4d_0213_{torch_fallback_kernels.dtype_names_dict[input.dtype]}")
    allocate_workspace_func(
        hidden_dim,
        heads,
        seq,
        batch_size,
        1,  # num_layers
        1,  # mp_size
        False,  # external_cache
        0,  # rank
        1024 * 100,  # max_out_tokens
        1)  # min_out_tokens
    return kernel_func(input, seq)


@pytest.mark.inference_ops
@pytest.mark.parametrize("inference_module", get_inference_modules())
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 12])
@pytest.mark.parametrize("sequence", [1, 18, 128])
@pytest.mark.parametrize("head_dim", [8, 64, 256, 512])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_transform4d_0213(inference_module, batch_size, num_heads, sequence, head_dim, dtype):
    activations_ds = torch.randn((batch_size, num_heads, sequence * head_dim),
                                 dtype=dtype,
                                 device=get_accelerator().device_name())
    activations_ref = activations_ds.clone().detach()
    ds_out = run_transform4d_0213_ds(inference_module, activations_ds, sequence)
    ref_out = run_transform4d_0213_reference(activations_ref, sequence)
    assert allclose(ds_out, ref_out)
