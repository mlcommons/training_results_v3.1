# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright Habana Labs, Ltd. an Intel Company'''

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from .inference_test_utils import allclose, get_dtypes
import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytestmark = pytest.mark.skip(reason="Inference ops are not available on this system")

inference_module = None


def run_bias_add_transform_0213_reference(input, num_heads, bias, trans_count):
    return torch_fallback_kernels.bias_add_transform_0213(input, bias, num_heads, trans_count)


def run_bias_add_transform_0213_ds(input, num_heads, bias, trans_count):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    batch_size = input.shape[0]
    seq_length = input.shape[1]
    value_size = input.shape[2]
    hidden_dim = value_size // trans_count
    head_dim = hidden_dim // num_heads

    # Resetting workspace, as when trans_count < 3, not all elements are currently filled in the kernel.
    inference_module.release_workspace()
    allocate_workspace_func = getattr(inference_module,
                                      f"allocate_workspace_{torch_fallback_kernels.dtype_names_dict[input.dtype]}")
    kernel_func = getattr(inference_module,
                          f"bias_add_transform_0213_{torch_fallback_kernels.dtype_names_dict[input.dtype]}")
    allocate_workspace_func(
        3 * hidden_dim,
        3 * num_heads,
        3 * seq_length,
        3 * batch_size,
        1,  # num_layers
        1,  # mp_size
        False,  # external_cache
        0,  # rank
        1024 * 100,  # max_out_tokens
        1)  # min_out_tokens
    return kernel_func(input, bias, num_heads, trans_count)


@pytest.mark.inference_ops
@pytest.mark.parametrize("trans_count", [1, 3])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 9, 18])
@pytest.mark.parametrize("value", [576, 1152, 2304])
@pytest.mark.parametrize("heads", [1, 12])
@pytest.mark.parametrize("dtype", get_dtypes())
def test_bias_add_transform_0213(trans_count, batch, sequence, value, heads, dtype):
    activations_ds = torch.randn((batch, sequence, value), dtype=dtype, device=get_accelerator().device_name())
    bias_ds = torch.randn((batch, sequence, value), dtype=dtype, device=get_accelerator().device_name())
    activations_ref = activations_ds.clone().detach()
    bias_ref = bias_ds.clone().detach()

    ds_out = run_bias_add_transform_0213_ds(activations_ds, heads, bias_ds, trans_count)
    ref_out = run_bias_add_transform_0213_reference(activations_ref, heads, bias_ref, trans_count)

    for i, (ds, ref) in enumerate(zip(ds_out, ref_out)):
        delta = ref - ds
        assert allclose(ds, ref), f"Tensor {i} {delta.max()=}, {delta.mean()=} {ds=}, {ref=}"
