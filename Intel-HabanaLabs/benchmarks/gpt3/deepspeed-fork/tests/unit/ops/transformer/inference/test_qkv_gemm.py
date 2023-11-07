# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import InferenceBuilder
from deepspeed.ops.op_builder.torch_fallback_builder import TorchInferenceOpBuilder
from .inference_test_utils import allclose
import torch
import torch.nn.functional as F

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytest.skip("Inference ops are not available on this system", allow_module_level=True)

inference_module = None
inference_torch_module = None


def run_qkv_gemm(inference_module, dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8,
                 transposed_mode):
    if dtype in [torch.float16, torch.int8]:
        inference_module.allocate_workspace_fp16(100, 100, 100, 200, 300000, 300000, False, 100, 100, 100)
        out, norm = inference_module.qkv_gemm_fp16(input, weight, q_scale, bias, gamma, beta, epsilon, add_bias,
                                                   q_int8, transposed_mode)
    elif dtype == torch.bfloat16:
        inference_module.allocate_workspace_bf16(100, 100, 100, 200, 300000, 300000, False, 100, 100, 100)
        out, norm = inference_module.qkv_gemm_bf16(input, weight, q_scale, bias, gamma, beta, epsilon, add_bias,
                                                   q_int8, transposed_mode)
    else:
        inference_module.allocate_workspace_fp32(100, 100, 100, 200, 300000, 300000, False, 100, 100, 100)
        out, norm = inference_module.qkv_gemm_fp32(input, weight, q_scale, bias, gamma, beta, epsilon, add_bias,
                                                   q_int8, transposed_mode)
    return [out, norm]


def run_qkv_gemm_ds(dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8, transposed_mode):
    global inference_module
    if inference_module is None:
        inference_module = InferenceBuilder().load()
    return run_qkv_gemm(inference_module, dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8,
                        transposed_mode)


def run_qkv_gemm_torch(dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8, transposed_mode):
    global inference_torch_module
    if inference_torch_module is None:
        inference_torch_module = TorchInferenceOpBuilder().load()
    return run_qkv_gemm(inference_torch_module, dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias,
                        q_int8, transposed_mode)


def run_qkv_gemm_reference(input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
    input = input.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)
    gamma.to(torch.float32)
    beta.to(torch.float32)
    inp_norm = F.layer_norm(input, (input.shape[2], ), gamma, beta, eps)
    tmp = torch.matmul(inp_norm, weight.t() if transpose else weight)
    if add_bias:
        tmp += bias
    output = [tmp, inp_norm]
    return output


@pytest.mark.inference_ops
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("dtype", [torch.float])
def test_qkv_gemm(transposed, dtype):
    input = torch.ones([1, 256, 1024], dtype=dtype, device=get_accelerator().device_name()).clone().detach()
    weight = torch.ones([3072, 1024], dtype=dtype, device=get_accelerator().device_name()).clone().detach()
    if not transposed:
        weight = weight.t()
    bias = torch.ones([3072], dtype=dtype, device=get_accelerator().device_name()).clone().detach()

    add_bias = bias is not None
    bias = bias if add_bias else torch.empty(1)  # type: ignore
    q_scale = weight.scale if hasattr(weight, 'scale') else torch.empty(
        1, device=get_accelerator().device_name())  # type: ignore
    q_int8 = dtype == torch.int8
    gamma = torch.ones(1024, device=get_accelerator().device_name())
    epsilon = 0.00001
    beta = torch.ones(1024, device=get_accelerator().device_name())

    ds_out = run_qkv_gemm_ds(dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8, transposed)
    torch_out = run_qkv_gemm_torch(dtype, input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8,
                                   transposed)
    ref_out = run_qkv_gemm_reference(input, weight, q_scale, bias, gamma, beta, epsilon, add_bias, q_int8, transposed)
    assert (allclose(ds_out[0], ref_out[0]))
    assert (allclose(ds_out[1], ref_out[1]))
    assert (allclose(torch_out[0], ref_out[0]))
    assert (allclose(torch_out[1], ref_out[1]))
