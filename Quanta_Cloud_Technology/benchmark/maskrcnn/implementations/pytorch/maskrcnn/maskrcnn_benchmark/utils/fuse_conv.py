# Copyright (c) 2018-2023 NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn
#from apex.contrib.conv_bias_relu import ConvBiasReLU, ConvBias
import fused_conv_bias_relu


class ConvBias_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        x = x.permute([0,3,1,2]).contiguous(memory_format=torch.channels_last) # convert to native nhwc
        outputs = fused_conv_bias_relu.forward_no_relu([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight)
        ctx.padding = padding
        ctx.stride = stride
        return outputs[0].permute([0,2,3,1]) # convert to explicit nhwc

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.permute([0,3,1,2])  # convert to native nhwc
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward_no_relu(bwd_args, padding, stride)
        return grads[0].permute([0,2,3,1]), grads[1].contiguous(memory_format=torch.channels_last), grads[2].contiguous(memory_format=torch.torch.channels_last), None, None

class ConvBiasReLU_(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.half)
    def forward(ctx, x, weight, bias, padding, stride):
        outputs = fused_conv_bias_relu.forward([x, weight, bias], padding, stride)
        ctx.save_for_backward(x, weight, outputs[0])
        ctx.padding = padding
        ctx.stride = stride
        return outputs[0]

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        bwd_args = [*ctx.saved_tensors, grad_output]
        padding = ctx.padding
        stride = ctx.stride
        grads = fused_conv_bias_relu.backward(bwd_args, padding, stride)
        return grads[0], grads[1], grads[2].reshape([-1]), None, None
