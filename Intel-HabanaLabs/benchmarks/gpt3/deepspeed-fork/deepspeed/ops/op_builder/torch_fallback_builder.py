# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# taken from op_builder/cpu/builder.py
try:
    # is op_builder from deepspeed or a 3p version? this should only succeed if it's deepspeed
    # if successful this also means we're doing a local install and not JIT compile path
    from op_builder import __deepspeed__  # noqa: F401
    from op_builder.builder import OpBuilder
    import op_builder.torch_fallback_kernels as torch_fallback_kernels
except ImportError:
    from deepspeed.ops.op_builder.builder import OpBuilder
    import deepspeed.ops.op_builder.torch_fallback_kernels as torch_fallback_kernels


class TorchInferenceOp:
    """Torch implementations for inference operations"""

    def gated_activation(activation, bias, activation_func_type):
        return torch_fallback_kernels.gated_activation_fallback(activation, bias, activation_func_type)

    def layer_norm(vals, gamma, beta, epsilon):
        return torch_fallback_kernels.layer_norm_fallback(vals, gamma, beta, epsilon)

    def _layer_norm_residual(vals, bias, res, gamma, beta, epsilon):
        return torch_fallback_kernels.layer_norm_residual_fallback(vals, bias, res, gamma, beta, epsilon)

    def layer_norm_residual_store_pre_ln_res(vals, bias, res, gamma, beta, epsilon):
        return torch_fallback_kernels.layer_norm_residual_store_pre_ln_res_fallback(
            vals, bias, res, gamma, beta, epsilon)

    def moe_res_matmul(residual, coef, output):
        return torch_fallback_kernels.moe_res_matmul_fallback(residual, coef, output)

    def reset_cache():
        """Nothing to do here"""

    def release_workspace():
        """No release necessary in Torch"""

    def retake_workspace():
        """Nothing necessary in Torch"""

    def pre_rms_norm(vals, residual, gamma, epsilon):
        return torch_fallback_kernels.pre_rms_norm_fallback(vals, residual, gamma, epsilon)

    def rms_norm(vals, gamma, epsilon):
        return torch_fallback_kernels.rms_norm_fallback(vals, gamma, epsilon)

    def _vector_add(a, b, gamma):
        return torch_fallback_kernels.vector_add_fallback(a, b, gamma)


def define_func_dtypes(clas, func, name=None):
    for dtype in torch_fallback_kernels.dtype_names_dict.values():
        setattr(clas, f"{name or func.__name__}_{dtype}", func)


def define_dtype_funcs_to_class(clas):
    define_func_dtypes(clas, torch_fallback_kernels.allocate_workspace)
    define_func_dtypes(clas, torch_fallback_kernels.bias_add)
    define_func_dtypes(clas, torch_fallback_kernels.bias_gelu_fallback, "bias_gelu")
    define_func_dtypes(clas, torch_fallback_kernels.bias_relu_fallback, "bias_relu")
    define_func_dtypes(clas, torch_fallback_kernels.gelu_gemm_fallback, "fused_gemm_gelu")
    define_func_dtypes(clas, torch_fallback_kernels.mlp_gemm_fallback, "mlp_gemm")
    define_func_dtypes(clas, torch_fallback_kernels.rms_mlp_gemm_fallback, "rms_mlp_gemm")
    define_func_dtypes(clas, torch_fallback_kernels.residual_add_bias_fallback, "residual_add_bias")
    define_func_dtypes(clas, torch_fallback_kernels.qkv_gemm_fallback, "qkv_gemm")
    define_func_dtypes(clas, torch_fallback_kernels.rms_qkv_gemm_fallback, "rms_qkv_gemm")
    define_func_dtypes(clas, torch_fallback_kernels.transform4d_0213, "transform4d_0213")
    define_func_dtypes(clas, torch_fallback_kernels.bias_add_transform_0213, "bias_add_transform_0213")
    define_func_dtypes(clas, torch_fallback_kernels.vector_matmul_fallback, "vector_matmul")
    define_func_dtypes(clas, torch_fallback_kernels.softmax_fallback, "softmax")
    define_func_dtypes(clas, torch_fallback_kernels.softmax_context_fallback, "softmax_context")


class TorchInferenceOpBuilder(OpBuilder):
    NAME = "transformer_inference"

    def __init__(self, name=None):
        name = self.NAME if name is None else name
        super().__init__(name=name)

    def absolute_name(self):
        return f"deepspeed.ops.transformer.inference.{self.NAME}_op"

    def sources(self):
        return []

    def load(self, verbose=True):
        clas = TorchInferenceOp
        define_dtype_funcs_to_class(clas)
        return clas
