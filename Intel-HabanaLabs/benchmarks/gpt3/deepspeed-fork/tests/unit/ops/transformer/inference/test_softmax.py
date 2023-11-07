# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''Copyright Habana Labs, Ltd. an Intel Company'''

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.transformer.inference.op_binding import SoftmaxOp
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
@pytest.mark.parametrize("heads", [1, 12, 24])
@pytest.mark.parametrize("triangular", [False, True])
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("rand", [False, True])
def test_softmax(batch, sequence, value, heads, triangular, dtype, rand):
    global inference_module
    if pkg_version.parse(torch.__version__) < pkg_version.parse("1.12"):
        pytest.skip("softmax implementation matches only after torch 1.12")

    ds_inference_config = DeepSpeedInferenceConfig()
    ds_inference_config.dtype = dtype
    softmax_op = SoftmaxOp(ds_inference_config)
    device_name = get_accelerator().device_name()

    alibi = torch.tensor([1], dtype=dtype, device=device_name)
    if (rand):
        torch.manual_seed(234)
        attn_scores = torch.randn((batch, heads, sequence, value), dtype=dtype, device=device_name)
        attn_mask = torch.randn((batch, value), dtype=dtype, device=device_name)
    else:
        attn_scores = torch.ones((batch, heads, sequence, value), dtype=dtype, device=device_name)
        attn_scores *= torch.tensor(0.1, dtype=dtype)
        attn_mask = torch.ones((batch, value), dtype=dtype, device=device_name)
        attn_mask *= torch.tensor(0.1, dtype=dtype)
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    inference_module.reset_cache()
    allocate_workspace_func = getattr(inference_module,
                                      f"allocate_workspace_{torch_fallback_kernels.dtype_names_dict[dtype]}")
    allocate_workspace_func(
        value,
        heads,
        sequence,
        batch,
        1,  # num_layers
        1,  # mp_size
        False,  # external_cache
        0,  # rank
        1024 * 100,  # max_out_tokens
        1)  # min_out_tokens)

    recompute = False
    local_attention = False
    window_size = 256
    async_op = False
    layer_scale = 1
    head_offset = 0
    mp_size = 1
    attn_scores_ref = attn_scores.clone().detach()
    attn_mask_ref = attn_mask.clone().detach()
    alibi_ref = alibi.clone().detach()

    ds_output = softmax_op.forward(attn_scores, attn_mask, alibi, triangular, recompute, local_attention, window_size,
                                   async_op, layer_scale, head_offset)

    fallback_output = torch_fallback_kernels.softmax_fallback(attn_scores_ref, attn_mask_ref, alibi_ref, triangular,
                                                              recompute, local_attention, window_size, async_op,
                                                              layer_scale, head_offset, mp_size)

    assert (allclose(ds_output, fallback_output))
    inference_module.release_workspace()
