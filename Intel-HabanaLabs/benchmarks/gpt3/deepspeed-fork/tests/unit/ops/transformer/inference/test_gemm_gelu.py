# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Copyright Habana Labs, Ltd. an Intel Company"""

import pytest
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.transformer.inference.op_binding import GELUGemmOp
from deepspeed.inference.config import DeepSpeedInferenceConfig
from deepspeed.ops.op_builder import InferenceBuilder
import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels
from .inference_test_utils import allclose, get_dtypes
from packaging import version as pkg_version

if not deepspeed.ops.__compatible_ops__[InferenceBuilder.NAME]:
    pytestmark = pytest.mark.skip(reason="Inference ops are not available on this system")


@pytest.mark.inference_ops
@pytest.mark.parametrize("dtype", get_dtypes())
@pytest.mark.parametrize("rand", [False, True])
def test_gemm_gelu(dtype, rand):
    if pkg_version.parse(torch.__version__) < pkg_version.parse("1.12"):
        pytest.skip("gemm gelu implementation matches only after torch 1.12")

    ds_inference_config = DeepSpeedInferenceConfig({})
    ds_inference_config.dtype = dtype
    gelu_gem_op = GELUGemmOp(ds_inference_config)
    device_name = get_accelerator().device_name()
    if (rand):
        input = torch.randn((3, 2, 4), dtype=dtype, device=device_name)
        weight1 = torch.randn((4, 16), dtype=dtype, device=device_name)
        bias = torch.randn((16), dtype=dtype, device=device_name)
        weight2 = torch.randn((16, 5), dtype=dtype, device=device_name)
    else:
        input = torch.ones((3, 2, 4), dtype=dtype, device=device_name)
        weight1 = torch.ones((4, 16), dtype=dtype, device=device_name)
        bias = torch.ones((16), dtype=dtype, device=device_name)
        weight2 = torch.ones((16, 5), dtype=dtype, device=device_name)
        expected = torch.ones((3, 2, 5), dtype=dtype, device=device_name) * 80

    output = gelu_gem_op.forward(input, weight1, bias, weight2)
    fallback_output = torch_fallback_kernels.gelu_gemm_fallback(input, weight1, None, bias, weight2, None, None, False)

    assert (allclose(output, fallback_output))
    if (not rand):
        assert (allclose(output, expected))
