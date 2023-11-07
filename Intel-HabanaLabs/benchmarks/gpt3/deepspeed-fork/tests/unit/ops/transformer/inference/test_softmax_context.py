# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright Habana Labs, Ltd. an Intel Company'''

import pytest
import torch
import math
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.transformer.inference.op_binding import SoftmaxContextOp
from deepspeed.ops.transformer.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.op_builder import InferenceBuilder
import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels
from .inference_test_utils import allclose, get_dtypes
from packaging import version as pkg_version

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytestmark = pytest.mark.skip(reason="Inference ops are not available on this system")

inference_module = None


@pytest.mark.inference_ops
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence", [1, 9, 18])
@pytest.mark.parametrize("value", [576, 1152, 2304])
@pytest.mark.parametrize("heads", [6, 12, 24])
@pytest.mark.parametrize("no_masking", [False, True])
@pytest.mark.parametrize("num_layers", [1, 2, 6])
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("rand", [False, True])
def test_softmax_context(batch, sequence, value, heads, no_masking, num_layers, dtype, rand):
    global inference_module
    if pkg_version.parse(torch.__version__) < pkg_version.parse("1.12"):
        pytest.skip("softmax_context implementation matches only after torch 1.12")

    ds_inference_config = DeepSpeedInferenceConfig()
    ds_inference_config.dtype = dtype
    softmax_context_op = SoftmaxContextOp(ds_inference_config)
    device_name = get_accelerator().device_name()
    norm_factor = math.sqrt((value // 3) // heads)

    alibi = None
    if (rand):
        torch.manual_seed(234)
        query_key_value = torch.randn((batch, sequence, value), dtype=dtype, device=device_name)
        query_key_value *= torch.tensor(0.1, dtype=dtype)
        attn_mask = torch.randn((batch, sequence), dtype=dtype, device=device_name)
        attn_mask *= torch.tensor(0.1, dtype=dtype)
        from random import randrange
        layer_id = randrange(num_layers)
    else:
        query_key_value = torch.ones((batch, sequence, value), dtype=dtype, device=device_name)
        query_key_value *= torch.tensor(0.1, dtype=dtype)
        attn_mask = torch.ones((batch, sequence), dtype=dtype, device=device_name)
        attn_mask *= torch.tensor(0.1, dtype=dtype)
        layer_id = 0
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    inference_module.reset_cache()
    allocate_workspace_func = getattr(inference_module,
                                      f"allocate_workspace_{torch_fallback_kernels.dtype_names_dict[dtype]}")
    allocate_workspace_func(
        value // 3,
        heads,
        sequence,
        batch,
        num_layers,  # num_layers
        1,  # mp_size
        False,  # external_cache
        0,  # rank
        1024 * 100,  # max_out_tokens
        1)  # min_out_tokens)
    query_key_value_ref = query_key_value.clone().detach()
    attn_mask_ref = attn_mask.clone().detach()

    output_q, output_k, output_v = softmax_context_op.forward(query_key_value, attn_mask, heads, norm_factor,
                                                              no_masking, layer_id, num_layers, alibi)

    fallback_output_q, fallback_output_k, fallback_output_v = torch_fallback_kernels.softmax_context_fallback(
        query_key_value_ref, attn_mask_ref, ds_inference_config.rotary_dim, ds_inference_config.rotate_half,
        ds_inference_config.rotate_every_two, heads, norm_factor, ds_inference_config.triangular_masking,
        ds_inference_config.local_attention, ds_inference_config.window_size, no_masking, layer_id, num_layers, alibi)

    assert (allclose(output_q, fallback_output_q))
    assert (allclose(output_k, fallback_output_k))
    assert (allclose(output_v, fallback_output_v))
    inference_module.release_workspace()
