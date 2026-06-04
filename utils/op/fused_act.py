# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# ---------------------------------------------------------------

""" Originated from https://github.com/rosinality/stylegan2-pytorch
The license for the original version of this file can be found in this directory (LICENSE_MIT).
"""

import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
# Fix: CUDA 12.8 + PyTorch 2.7 has breaking C++ API changes in the custom kernels.
# The native PyTorch implementation (F.leaky_relu) is used instead via fused_leaky_relu(),
# so we wrap the compilation in try-except to allow graceful fallback.
try:
    fused = load(
        "fused",
        sources=[
            os.path.join(module_path, "fused_bias_act.cpp"),
            os.path.join(module_path, "fused_bias_act_kernel.cu"),
        ],
        extra_cuda_cflags=['-allow-unsupported-compiler'],
    )
except Exception as e:
    print(f"[SynDiff] Custom CUDA kernel 'fused' failed to compile: {e}")
    print("[SynDiff] Using pure PyTorch fallback (F.leaky_relu) instead.")
    # Create a dummy module to avoid import errors
    from types import SimpleNamespace
    fused = SimpleNamespace()
    fused.fused_bias_act = None


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    # Fix: Use pure PyTorch implementation uniformly
    # The custom CUDA kernel (FusedLeakyReLUFunction) fails to compile with
    # CUDA 12.8 + GCC 14.2. The native PyTorch path is GPU-accelerated via
    # F.leaky_relu and is equivalent in functionality.
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    return (
        F.leaky_relu(
            input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
        )
        * scale
    )
