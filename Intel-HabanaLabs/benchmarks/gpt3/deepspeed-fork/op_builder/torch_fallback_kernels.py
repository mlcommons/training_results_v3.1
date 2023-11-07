# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Copyright Habana Labs, Ltd. an Intel Company"""

import torch
import torch.nn.functional as F

dtype_names_dict = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float32: "fp32",
}


def allocate_workspace(hidden_dim, num_heads, prompt_length, batch_size, num_layers, mp_size, external_cache, rank,
                       max_out_tokens, min_out_tokens):
    """No allocation necessary in Torch"""


def bias_add(input, bias):
    return torch.add(input, bias)


def bias_gelu_fallback(activations, bias):
    # Expected behavior is that of casting to float32 internally and using the tanh approximation
    return F.gelu(activations.to(torch.float32) + bias.to(torch.float32), approximate='tanh').to(activations.dtype)


def bias_relu_fallback(activations, bias):
    # Expected behavior is that of casting to float32 internally
    return F.relu(activations.to(torch.float32) + bias.to(torch.float32)).to(activations.dtype)


def gated_geglu_fallback(activations, bias):
    # Expected behavior is that of casting to float32 internally
    # Explicitly using the default GeLU
    activations = activations + bias.reshape(1, 1, -1)
    hidden_states, gate = activations.chunk(2, dim=-1)
    return hidden_states * F.gelu(gate.to(torch.float32)).to(activations.dtype)


def gated_silu_fallback(activations, bias):
    # Expected behavior is that of casting to float32 internally
    # Explicitly using the default GeLU
    activations = activations + bias.reshape(1, 1, -1)
    hidden_states, gate = activations.chunk(2, dim=-1)
    return hidden_states * F.silu(gate.to(torch.float32)).to(activations.dtype)


def gated_activation_fallback(activations, bias, activation_func_type):
    from deepspeed.utils.types import ActivationFuncType
    if activation_func_type == ActivationFuncType.GATED_SILU:
        return gated_silu_fallback(activations, bias)
    elif activation_func_type == ActivationFuncType.GATED_GELU:
        return gated_geglu_fallback(activations, bias)
    # Default, shouldn't happen
    raise NotImplementedError


def gelu_gemm_fallback(input, weight, scale, bias, out, out_scale, dtype, transpose):
    tmp = torch.matmul(input, weight)
    tmp = F.gelu(tmp.to(torch.float32) + bias.to(torch.float32), approximate="tanh").to(tmp.dtype)
    output = torch.matmul(tmp, out)
    return output


def layer_norm_fallback(vals, gamma, beta, epsilon):
    channels = gamma.shape[0]
    dtype = gamma.dtype
    vals_f = vals.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return F.layer_norm(vals_f, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)


def layer_norm_residual_fallback(vals, bias, res, gamma, beta, epsilon):
    channels = gamma.shape[0]
    dtype = gamma.dtype
    vals_f = vals.to(torch.float32)
    bias_f = bias.to(torch.float32).reshape(1, 1, -1)
    res_f = res.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    return F.layer_norm(vals_f + bias_f + res_f, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)


def layer_norm_residual_store_pre_ln_res_fallback(vals, bias, res, gamma, beta, epsilon):
    channels = gamma.shape[0]
    dtype = gamma.dtype
    vals_f = vals.to(torch.float32)
    bias_f = bias.to(torch.float32).reshape(1, 1, -1)
    res_f = res.to(torch.float32)
    gamma_f = gamma.to(torch.float32)
    beta_f = beta.to(torch.float32)
    res_output = vals_f + bias_f + res_f
    norm_output = F.layer_norm(res_output, (channels, ), weight=gamma_f, bias=beta_f, eps=epsilon).to(dtype)
    return norm_output, res_output.to(dtype)


def moe_res_matmul_fallback(residual, coef_t, output):
    coef = coef_t.transpose(1, 2).contiguous()
    coef1, coef2 = torch.split(coef, split_size_or_sections=coef.shape[len(coef.shape) - 1] // 2, dim=-1)
    return residual * coef1 + output * coef2


def mlp_gemm_fallback(
    input,
    residual,
    input_bias,
    weight_interm,
    weight_out,
    bias,
    gamma,
    beta,
    eps,
    pre_layer_norm,
    mlp_after_attn,
    interm_scale,
    out_scale,
    dtype,
    mlp_act_func_type,
    transpose,
):
    if mlp_after_attn:
        residual_add = F.layer_norm(
            input + residual + input_bias,
            (input.shape[2], ),
            gamma,
            beta,
            eps,
        )
        tmp = torch.matmul(residual_add, weight_interm.t() if transpose else weight_interm)
        tmp = F.gelu(tmp + bias)
        output = torch.matmul(tmp, weight_out.t() if transpose else weight_out)
        return (output, residual_add)
    else:
        # TODO: SW-151870 implement mlp_gemm_fallback
        raise NotImplementedError


def rms_mlp_gemm_fallback(
    input,
    residual,
    weight_interm,
    weight_out,
    gamma,
    eps,
    interm_scale,
    out_scale,
    dtype,
    mlp_act_func_type,
    transpose,
):
    # TODO: SW-151864 implement rms_mlp_gemm_fallback
    raise NotImplementedError


def pre_rms_norm_fallback(vals, residual, gamma, epsilon):
    residual = vals.to(torch.float32) + residual.to(torch.float32)
    vals = residual

    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals, residual.to(gamma.dtype)


def rms_norm_fallback(vals, gamma, epsilon):
    variance = vals.to(torch.float32).pow(2).mean(-1, keepdim=True)
    vals = vals * torch.rsqrt(variance + epsilon)

    if gamma.dtype in [torch.float16, torch.bfloat16]:
        vals = vals.to(gamma.dtype)

    return gamma * vals


def res_add_bias_ref_gptj_fallback(hidden_state, residual, attn_output, attn_bias, final_bias, add_attn_bias, mp_size):
    hidden_state += attn_output + (residual + final_bias) / mp_size
    if add_attn_bias:
        hidden_state += attn_bias / mp_size
    return hidden_state


def residual_add_bias_fallback(
    hidden_state,
    residual,
    attention_output,
    attention_bias,
    final_bias,
    mp_size,
    mlp_after_attn,
    add_bias,
    pre_layer_norm,
):
    if mlp_after_attn:
        if pre_layer_norm:
            tmp = (residual.float() + attention_output.float() + attention_bias.float() +
                   final_bias.float()) / mp_size + hidden_state.float()
        else:
            tmp = residual.float() + hidden_state.float() + final_bias.float()

        input_dtype = hidden_state.dtype
        residual = tmp.to(input_dtype)
        return residual
    else:
        return res_add_bias_ref_gptj_fallback(hidden_state, residual, attention_output, attention_bias, final_bias,
                                              add_bias, mp_size)


def qkv_gemm_fallback(input, weight, q_scale, bias, gamma, beta, eps, add_bias, q_int8, transpose):
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


def rms_qkv_gemm_fallback(input, weight, q_scale, gamma, eps, q_int8, transpose):
    # TODO: SW-151624 implement this qkv_gemm case
    raise NotImplementedError


def softmax_fallback(
    attn_scores,
    attn_mask,
    alibi,
    triangular,
    recompute,
    local_attention,
    window_size,
    async_op,
    layer_scale,
    head_offset,
    mp_size,
):
    # get heads, algo from kernel code
    len_ = len(attn_scores.size())
    heads = 1
    if len_ > 1:
        heads = attn_scores.size()[1]
    num_attention_heads_per_partition = heads // mp_size

    if alibi != None:
        if len(alibi.shape) == 1:
            alibi = None
        else:
            alibi = alibi[head_offset:head_offset + num_attention_heads_per_partition]
    if attn_mask != None and len(attn_mask.shape) == 1:
        attn_mask = None
    input_dtype = attn_scores.dtype
    attn_scores *= layer_scale
    if triangular:
        tri = ~torch.tril(torch.ones(attn_scores.size(), device=attn_scores.device)).to(bool)
        attn_scores = torch.masked_fill(attn_scores, tri, torch.finfo(input_dtype).min)
    if alibi is not None:
        attn_scores += alibi
    if attn_mask is not None:
        # expand atten_mask from two dim into 4 dim, insert two dims in the middle
        if len(attn_mask.shape) == 2:
            # The above if statement was added because the mask was already 4D so this
            # expansion should be avoided as it expands to 6D and crashes later (in bloom
            # HE KI FB)
            attn_mask = attn_mask[:, None, None, :]
        attn_scores += attn_mask
    output = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(input_dtype)
    return output


def transform4d_0213(x, seq_length):
    assert x.dim() == 3, F"{x.dim()=} is not supported"
    batch_size, num_heads, seq_length_head_dim = x.shape
    head_dim = seq_length_head_dim // seq_length
    x = x.view(batch_size, num_heads, seq_length, head_dim)
    x = x.permute(0, 2, 1, 3)
    return x


def bias_add_transform_0213(input, bias, num_heads, trans_count, perform_bias=False):
    assert trans_count == 1 or trans_count == 3, F"{trans_count=} is not supported"
    assert input.dim() == 3, F"{input.dim()=} is not supported"
    input_biased = bias_add(input, bias) if perform_bias else input
    batch_size, seq_length, value_size = input_biased.shape
    hid_dim = value_size // trans_count
    head_dim = hid_dim // num_heads

    if (trans_count == 1):
        query_layer = input.view(batch_size, seq_length, num_heads, head_dim)
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = torch.zeros_like(query_layer)
        value_layer = torch.zeros_like(query_layer)
        return query_layer, key_layer, value_layer

    qkv_layers = input.view(batch_size, seq_length, 3, num_heads, head_dim)
    query_layer, key_layer, value_layer = qkv_layers[..., 0, :, :], qkv_layers[..., 1, :, :], qkv_layers[..., 2, :, :]
    query_layer = query_layer.transpose(1, 2)
    key_layer = key_layer.transpose(1, 2)
    value_layer = value_layer.transpose(1, 2)
    return query_layer, key_layer, value_layer


def vector_matmul_fallback(input, weight, async_op, q_scale, q_int8, transpose):
    return torch.matmul(input, weight.t() if transpose else weight)


def vector_add_fallback(a, b, gamma):
    """Based on csrc/transformer/inference/csrc/pt_binding.cpp code of _vector_add"""
    dtype = a.dtype
    return (gamma * a.float() + b.float()).to(dtype)


def softmax_context_fallback(query_key_value, attn_mask, rotary_dim, rotate_half, roteate_every_two, heads,
                             norm_factor, triangular_masking, local_attention, window_size, no_masking, layer_id,
                             num_layers, alibi):
    bat_0213_query, bat_0213_key, bat_0213_value = bias_add_transform_0213(query_key_value, None, heads, 3, False)

    #if (rotary_dim > 0 and rotate_half):
    #    TODO: Add implementation for this case

    bsz = query_key_value.shape[0]
    seq_len = query_key_value.shape[1]
    head_dim = query_key_value.shape[2] // (heads * 3)
    bmm_output = torch.bmm(bat_0213_query.reshape(bsz * heads, seq_len, head_dim),
                           bat_0213_key.reshape(bsz * heads, seq_len, head_dim).transpose(1, 2))
    layer_scale = 1.0
    if alibi != None and len(alibi.shape) > 1:
        layer_scale = max(1, layer_id).to(float)

    alpha = norm_factor * norm_factor / layer_scale
    bmm_output *= alpha
    bmm_output_reshape = bmm_output.reshape(bsz, heads, seq_len, seq_len)

    recompute = seq_len > 1
    softmax_output = softmax_fallback(bmm_output_reshape, attn_mask, alibi, triangular_masking, recompute,
                                      local_attention, window_size, None, layer_scale, 0, 1)

    output = torch.bmm(softmax_output.reshape(bsz * heads, seq_len, seq_len),
                       bat_0213_value.reshape(bsz * heads, seq_len, head_dim))

    output = output.reshape(bsz, heads, seq_len, head_dim)
    output = output.reshape(bsz, heads, seq_len * head_dim)
    t4d_0123_output = transform4d_0213(output, seq_len)
    t4d_0123_output = t4d_0123_output.reshape(bsz, seq_len, heads * head_dim)

    return t4d_0123_output, bat_0213_key, bat_0213_value
