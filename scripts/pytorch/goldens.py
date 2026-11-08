"""Golden tensors from PyTorch for jitNN's operator tests.

Each case applies one operator to fixed random inputs, then back-propagates loss = sum(y * w)
with a random upstream gradient w. Inputs, the output and every input gradient are written to
jitnn-core/src/test/resources/golden/ as:

  manifest.txt   one line per array:  <case> <role> <dtype> <offset> <dims...>
                 and one line per case: case <name> <op> <attrs...>
  data.bin       the arrays, little-endian float32 / int32, back to back

Roles: in0, in1 ... (inputs), w (upstream gradient), out, grad0, grad1 ... (input gradients),
and for optimiser cases p0 (start), g<k> (gradient at step k), out (final parameter).

usage: python goldens.py <output dir>
"""
import os
import sys

import numpy as np
import torch

from jitnn_init import uniform

rng = np.random.default_rng(1234)
torch.set_default_dtype(torch.float64)  # compute goldens in FP64, store FP32

cases = []  # (name, op, attrs, inputs[list of np arrays], grads_for[list of bool])


def rnd(*shape, lo=-2.0, hi=2.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def pos(*shape):
    return rng.uniform(0.2, 3.0, size=shape).astype(np.float32)


def add(name, op, attrs, inputs, grads=None):
    cases.append((name, op, attrs, inputs, grads if grads is not None else [a.dtype == np.float32 for a in inputs]))


# ---- binary with broadcasting, odd and size-1 shapes
for op in ["add", "sub", "mul", "div", "maximum"]:
    b = (lambda *s: pos(*s)) if op == "div" else rnd
    add(f"{op}_same", op, [], [rnd(37, 5), b(37, 5)])
    add(f"{op}_row", op, [], [rnd(7, 33), b(33)])
    add(f"{op}_col", op, [], [rnd(7, 33), b(7, 1)])
    add(f"{op}_3d", op, [], [rnd(3, 1, 5), b(4, 1)])
    add(f"{op}_scalar", op, [], [rnd(1), b(9, 2)])
# ---- unary
for op, gen in [("neg", rnd), ("exp", rnd), ("log", pos), ("sqrt", pos), ("abs", rnd), ("relu", rnd), ("reciprocal", pos), ("square", rnd)]:
    add(f"{op}", op, [], [gen(17, 13)])
    add(f"{op}_1", op, [], [gen(1)])
add("pow_2.5", "pow", [2.5], [pos(11, 7)])
add("clamp", "clamp", [-0.7, 0.9], [rnd(11, 7)])
add("affine", "affine", [1.5, -0.25], [rnd(11, 7)])
# ---- reductions
for op in ["sum", "mean", "max", "min"]:
    add(f"{op}_ax0", op, [0, 0], [rnd(19, 23)])
    add(f"{op}_ax1_keep", op, [1, 1], [rnd(19, 23)])
    add(f"{op}_3d_mid", op, [1, 0], [rnd(3, 300, 5)])
    add(f"{op}_long", op, [0, 0], [rnd(5000)])
add("sum_all", "sum_all", [], [rnd(4, 5, 6)])
add("mean_all", "mean_all", [], [rnd(101)])
add("argmax", "argmax", [1], [rnd(13, 29)], [False])
# ---- matmul, all transposes, odd sizes and 1x1
for ta in (0, 1):
    for tb in (0, 1):
        for (m, n, k) in [(17, 33, 9), (1, 1, 1), (40, 3, 70)]:
            a = rnd(k, m) if ta else rnd(m, k)
            b = rnd(n, k) if tb else rnd(k, n)
            add(f"matmul_t{ta}{tb}_{m}x{n}x{k}", "matmul", [ta, tb], [a, b])
# ---- shape
add("permute_201", "permute", [2, 0, 1], [rnd(2, 3, 4)])
add("transpose", "permute", [1, 0], [rnd(9, 31)])
add("reshape", "reshape", [6, 4], [rnd(2, 3, 4)])
# ---- softmax family and losses
add("softmax", "softmax", [], [rnd(6, 10, lo=-5, hi=5)])
add("softmax_long", "softmax", [], [rnd(3, 1000, lo=-5, hi=5)])
add("log_softmax", "log_softmax", [], [rnd(6, 10, lo=-5, hi=5)])
add("cross_entropy", "cross_entropy", [], [rnd(16, 10, lo=-4, hi=4), rng.integers(0, 10, 16).astype(np.int32)])
add("linear", "linear", [], [rnd(8, 20), rnd(12, 20), rnd(12)])
add("linear_relu", "linear_relu", [], [rnd(8, 20), rnd(12, 20), rnd(12)])
add("linear_relu_big", "linear_relu", [], [rnd(128, 96), rnd(72, 96), rnd(72)])
add("linear_nobias", "linear_nobias", [], [rnd(16, 24), rnd(8, 24)])
# ---- convolution (attrs: stride pad dilation groups), batch norm, pooling, relu6
for (name, n, cin, h, w, cout, k, st, pd, dl, g) in [
    ("conv_3x3", 2, 3, 9, 11, 8, 3, 1, 1, 1, 1),
    ("conv_stride2", 2, 4, 10, 10, 6, 3, 2, 1, 1, 1),
    ("conv_7x7_s2", 1, 3, 17, 17, 5, 7, 2, 3, 1, 1),
    ("conv_1x1", 3, 16, 5, 5, 70, 1, 1, 0, 1, 1),
    ("conv_dilated", 1, 2, 12, 12, 3, 3, 1, 2, 2, 1),
    ("conv_groups", 2, 8, 7, 7, 12, 3, 1, 1, 1, 4),
    ("conv_depthwise", 2, 6, 8, 8, 6, 3, 2, 1, 1, 6),
    ("conv_big", 4, 32, 16, 16, 96, 3, 1, 1, 1, 1),
]:
    add(name, "conv2d", [st, pd, dl, g], [rnd(n, cin, h, w), rnd(cout, cin // g, k, k, lo=-0.5, hi=0.5), rnd(cout)])
add("conv_nobias", "conv2d_nobias", [1, 1, 1, 1], [rnd(2, 3, 6, 6), rnd(4, 3, 3, 3, lo=-0.5, hi=0.5)])
add("batch_norm", "batch_norm", [], [rnd(4, 5, 6, 7), rnd(5, lo=0.5, hi=1.5), rnd(5)])
add("batch_norm_2d", "batch_norm", [], [rnd(32, 9), rnd(9, lo=0.5, hi=1.5), rnd(9)])
add("max_pool_3s2p1", "max_pool2d", [3, 2, 1], [rnd(2, 3, 11, 11)])
add("max_pool_2s2", "max_pool2d", [2, 2, 0], [rnd(2, 4, 8, 8)])
add("avg_pool_3s2p1", "avg_pool2d", [3, 2, 1], [rnd(2, 3, 11, 11)])
add("avg_pool_2s2", "avg_pool2d", [2, 2, 0], [rnd(2, 4, 8, 8)])
add("global_avg_pool", "global_avg_pool", [], [rnd(3, 7, 5, 5)])
add("relu6", "relu6", [], [rnd(17, 13, lo=-3, hi=9)])
# ---- activations
for op in ["tanh", "sigmoid", "gelu", "gelu_tanh", "silu"]:
    add(op, op, [], [rnd(17, 13, lo=-6, hi=6)])
    add(f"{op}_tails", op, [], [rnd(64, lo=-12, hi=12)])
add("swiglu", "swiglu", [], [rnd(9, 14, lo=-4, hi=4), rnd(9, 14)])
# ---- slicing and joining (attrs: axis start length / axis)
add("narrow_mid", "narrow", [1, 2, 4], [rnd(4, 9, 5)])
add("narrow_first", "narrow", [0, 1, 2], [rnd(3, 7)])
add("narrow_last", "narrow", [2, 5, 3], [rnd(2, 3, 8)])
add("narrow_size1", "narrow", [1, 0, 1], [rnd(1, 6, 1)])
add("concat_mid", "concat", [1], [rnd(2, 3, 4), rnd(2, 1, 4), rnd(2, 5, 4)])
add("concat_first", "concat", [0], [rnd(3, 6), rnd(4, 6)])
add("concat_last", "concat", [-1], [rnd(5, 2), rnd(5, 7)])
add("stack_0", "stack", [0], [rnd(3, 4), rnd(3, 4), rnd(3, 4)])
add("stack_last", "stack", [2], [rnd(3, 4), rnd(3, 4)])
# ---- embedding (repeated, unused and padding rows); attrs: padding index or -1
add("embedding", "embedding", [-1], [rng.integers(0, 11, (3, 5)).astype(np.int32), rnd(11, 6)])
add("embedding_padding", "embedding", [2], [np.array([2, 0, 2, 5, 5, 5, 1], np.int32), rnd(7, 4)])
add("embedding_wide", "embedding", [-1], [rng.integers(0, 50, 300).astype(np.int32), rnd(50, 33)])
# ---- batched matmul (equal batches, or one side shared) and attention
for ta in (0, 1):
    for tb in (0, 1):
        a = rnd(2, 3, 7, 5) if ta else rnd(2, 3, 5, 7)
        b = rnd(2, 3, 4, 7) if tb else rnd(2, 3, 7, 4)
        add(f"bmm_t{ta}{tb}", "matmul", [ta, tb], [a, b])
add("bmm_shared_b", "matmul", [0, 0], [rnd(3, 5, 7), rnd(7, 4)])
add("bmm_shared_b_t", "matmul", [0, 1], [rnd(3, 5, 7), rnd(4, 7)])
add("bmm_shared_a", "matmul", [0, 0], [rnd(5, 7), rnd(3, 7, 4)])
add("bmm_big", "matmul", [0, 1], [rnd(4, 70, 32), rnd(4, 90, 32)])
# attrs: causal, scale (-1: 1/sqrt(E))
add("sdpa", "sdpa", [0, -1], [rnd(2, 3, 5, 8), rnd(2, 3, 7, 8), rnd(2, 3, 7, 6)])
add("sdpa_causal", "sdpa", [1, -1], [rnd(2, 2, 6, 8), rnd(2, 2, 6, 8), rnd(2, 2, 6, 8)])
add("sdpa_causal_rect", "sdpa", [1, 0.5], [rnd(1, 2, 4, 8), rnd(1, 2, 6, 8), rnd(1, 2, 6, 5)])
add("sdpa_long", "sdpa", [1, -1], [rnd(1, 2, 300, 16), rnd(1, 2, 300, 16), rnd(1, 2, 300, 16)])
add("sdpa_head128", "sdpa", [1, -1], [rnd(1, 2, 40, 128), rnd(1, 2, 40, 128), rnd(1, 2, 40, 128)])
add("sdpa_head96_mask", "sdpa_mask", [0, -1], [rnd(2, 1, 9, 96), rnd(2, 1, 13, 96), rnd(2, 1, 13, 96), np.where(rng.uniform(size=(2, 1, 9, 13)) < 0.2, -1e9, 0).astype(np.float32)], [True, True, True, False])
# [B, T, H, E] layouts read directly by the fused kernels; RoPE with the heads after the tokens
add("sdpa_bthd_causal", "sdpa_bthd", [1, -1], [rnd(2, 6, 3, 16), rnd(2, 6, 3, 16), rnd(2, 6, 3, 16)])
add("sdpa_bthd_mask", "sdpa_bthd_mask", [0, -1], [rnd(2, 5, 2, 8), rnd(2, 7, 2, 8), rnd(2, 7, 2, 8), np.where(rng.uniform(size=(2, 1, 1, 7)) < 0.3, -1e9, 0).astype(np.float32)], [True, True, True, False])
add("rope_heads", "rope_heads", [10000.0, 0, 0], [rnd(2, 5, 3, 8)])
add("rope_heads_interleaved", "rope_heads", [500000.0, 3, 1], [rnd(1, 4, 2, 16)])
pad = np.zeros((2, 1, 1, 7), np.float32)
pad[0, 0, 0, 5:] = -1e9
pad[1, 0, 0, 3:] = -1e9
add("sdpa_padding", "sdpa_mask", [0, -1], [rnd(2, 3, 5, 8), rnd(2, 3, 7, 8), rnd(2, 3, 7, 6), pad], [True, True, True, False])
# multi-head self-attention (attrs: heads causal); inputs x, Wqkv, bqkv, Wout, bout
for name, causal in [("mha", 0), ("mha_causal", 1)]:
    add(name, "mha", [4, causal], [rnd(2, 6, 16), rnd(48, 16, lo=-0.5, hi=0.5), rnd(48), rnd(16, 16, lo=-0.5, hi=0.5), rnd(16)])
add("mha_padding", "mha_mask", [4, 0], [rnd(2, 6, 16), rnd(48, 16, lo=-0.5, hi=0.5), rnd(48), rnd(16, 16, lo=-0.5, hi=0.5), rnd(16),
    np.array([[[[0, 0, 0, 0, -1e9, -1e9]]], [[[0, 0, 0, 0, 0, -1e9]]]], np.float32)], [True, True, True, True, True, False])
# rotary position embedding (attrs: base offset interleaved), computed directly in FP64
add("rope_half", "rope", [10000.0, 0, 0], [rnd(2, 3, 5, 8)])
add("rope_interleaved", "rope", [10000.0, 0, 1], [rnd(2, 3, 5, 8)])
add("rope_offset", "rope", [500000.0, 7, 0], [rnd(1, 2, 4, 16)])
add("rope_long", "rope", [10000.0, 0, 0], [rnd(1, 1, 300, 64)])
# top-k values (attrs: k); random inputs have no ties
add("topk", "topk", [3], [rnd(5, 11)])
add("topk_long", "topk", [8], [rnd(2, 3, 900)])
# half-precision products large enough for Metal's simdgroup matrices (M, N multiples of 8)
add("matmul_half_big_f16", "matmul_half", [0, 1, 0], [rnd(64, 48), rnd(96, 48)])
add("matmul_half_big_bf16", "matmul_half", [1, 0, 1], [rnd(40, 128), rnd(40, 72)])
add("linear_half_big", "linear_half", [0], [rnd(2, 64, 40), rnd(80, 40), rnd(80)])
# whole 128 x 128 tiles with B transposed: the copied tensor-core GEMM on CUDA
add("matmul_mma_f16", "matmul_half", [0, 1, 0], [rnd(128, 64), rnd(256, 64)])
add("matmul_mma_bf16", "matmul_half", [0, 1, 1], [rnd(256, 32), rnd(128, 32)])
add("linear_mma", "linear_half", [0], [rnd(2, 128, 48), rnd(128, 48), rnd(128)])
add("matmul_mma_nn", "matmul_half", [0, 0, 0], [rnd(128, 48), rnd(48, 256)])
add("matmul_mma_tn", "matmul_half", [1, 0, 1], [rnd(32, 256), rnd(32, 128)])
add("matmul_mma_tt", "matmul_half", [1, 1, 0], [rnd(64, 128), rnd(128, 64)])
add("matmul_mma_splitk", "matmul_half", [1, 0, 1], [rnd(2048, 128), rnd(2048, 128)])
# few 128 x 128 tiles and a long K: tensor-core split-K in the forward products
add("linear_mma_splitk", "linear_half", [0], [rnd(128, 512, lo=-1, hi=1), rnd(256, 512, lo=-1, hi=1), rnd(256)])
add("matmul_mma_splitk_nt", "matmul_half", [0, 1, 1], [rnd(128, 1024, lo=-1, hi=1), rnd(128, 1024, lo=-1, hi=1)])
# split-K (A^T B with a long reduction and a small output, as weight gradients)
add("matmul_splitk", "matmul", [1, 0], [rnd(1024, 40), rnd(1024, 24)])
add("matmul_splitk_odd", "matmul", [1, 0], [rnd(768, 70), rnd(768, 9)])
add("matmul_splitk_f16", "matmul_half", [1, 0, 0], [rnd(1024, 40), rnd(1024, 24)])
add("matmul_splitk_bf16", "matmul_half", [1, 0, 1], [rnd(1024, 40), rnd(1024, 24)])
# casts to FP16 / BF16 and back (round to nearest even); ties, subnormals and the extremes
ties = np.array([1 + 2**-11, 1 + 3 * 2**-11, -(1 + 2**-11), 1 + 2**-8, 1 + 3 * 2**-8, 2**-20, -3e-7, 6e-8, 65504.0, 60000.0, 1e-3, 0.0], np.float32)
add("cast_f16", "cast_f16", [], [np.concatenate([rnd(200, lo=-300, hi=300), ties])])
add("cast_bf16", "cast_bf16", [], [np.concatenate([rnd(200, lo=-300, hi=300), ties, np.array([3e38, -1e-30], np.float32)])])
# half-precision GEMM operands (attrs: ta tb type 0=f16 1=bf16): inputs and the incoming gradient
# rounded to that type, products exact (jitNN accumulates in FP32)
for kind in (0, 1):
    for ta in (0, 1):
        for tb in (0, 1):
            a = rnd(33, 17) if ta else rnd(17, 33)
            b = rnd(40, 33) if tb else rnd(33, 40)
            add(f"matmul_half{kind}_t{ta}{tb}", "matmul_half", [ta, tb, kind], [a, b])
    add(f"bmm_half{kind}", "matmul_half", [0, 1, kind], [rnd(2, 3, 9, 16), rnd(2, 3, 11, 16)])
    add(f"linear_half{kind}", "linear_half", [kind], [rnd(2, 7, 24), rnd(10, 24), rnd(10)])
# ---- LayerNorm and RMSNorm over the last axis; attrs: eps
add("layer_norm", "layer_norm", [1e-5], [rnd(7, 33), rnd(33, lo=0.5, hi=1.5), rnd(33)])
add("layer_norm_3d", "layer_norm", [1e-5], [rnd(2, 5, 48), rnd(48, lo=0.5, hi=1.5), rnd(48)])
add("layer_norm_long", "layer_norm", [1e-5], [rnd(3, 1000, lo=1, hi=5), rnd(1000, lo=0.5, hi=1.5), rnd(1000)])
add("layer_norm_rows", "layer_norm", [1e-5], [rnd(150, 16), rnd(16, lo=0.5, hi=1.5), rnd(16)])
add("layer_norm_nobias", "layer_norm_nobias", [1e-5], [rnd(6, 20), rnd(20, lo=0.5, hi=1.5)])
add("rms_norm", "rms_norm", [1e-6], [rnd(7, 33), rnd(33, lo=0.5, hi=1.5)])
add("rms_norm_3d", "rms_norm", [1e-6], [rnd(2, 5, 48), rnd(48, lo=0.5, hi=1.5)])
add("rms_norm_long", "rms_norm", [1e-6], [rnd(3, 1000), rnd(1000, lo=0.5, hi=1.5)])
add("rms_norm_rows", "rms_norm", [1e-6], [rnd(150, 16), rnd(16, lo=0.5, hi=1.5)])
# ---- activations and losses (attrs: slope)
add("leaky_relu", "leaky_relu", [0.01], [rnd(17, 13)])
add("leaky_relu_0.2", "leaky_relu", [0.2], [rnd(9, 4)])
add("softplus", "softplus", [], [rnd(17, 13)])
add("softplus_wide", "softplus", [], [rnd(9, 21, lo=-40, hi=40)])
add("mse", "mse", [], [rnd(7, 9), rnd(7, 9)])
add("mse_sum", "mse_sum", [], [rnd(3, 1, 5), rnd(3, 1, 5)])
p_edge = np.concatenate([rng.uniform(0.01, 0.99, size=40).astype(np.float32), np.array([0.0, 1.0, 1e-30, 1 - 2**-24], np.float32)])
add("bce", "bce", [], [p_edge.reshape(4, 11), rng.uniform(0, 1, size=(4, 11)).astype(np.float32)])
add("bce_hard", "bce", [], [rng.uniform(0.001, 0.999, size=(6, 7)).astype(np.float32), (rng.uniform(size=(6, 7)) < 0.5).astype(np.float32)])
add("bce_logits", "bce_logits", [], [rnd(5, 13, lo=-40, hi=40), rng.uniform(0, 1, size=(5, 13)).astype(np.float32)])
add("bce_logits_small", "bce_logits", [], [rnd(4, 9), (rng.uniform(size=(4, 9)) < 0.5).astype(np.float32)])
add("gaussian_kl", "gaussian_kl", [], [rnd(5, 8), rnd(5, 8)])
# ---- transposed convolution (attrs: stride pad output_padding dilation groups) and resizing
add("conv_t", "conv_transpose2d", [2, 1, 1, 1, 1], [rnd(2, 4, 5, 5), rnd(4, 3, 3, 3), rnd(3)])
add("conv_t_k4s2", "conv_transpose2d", [2, 1, 0, 1, 1], [rnd(2, 8, 4, 4), rnd(8, 4, 4, 4), rnd(4)])
add("conv_t_s1", "conv_transpose2d", [1, 0, 0, 1, 1], [rnd(1, 5, 4, 3), rnd(5, 2, 3, 3), rnd(2)])
add("conv_t_groups_dil", "conv_transpose2d", [1, 0, 0, 2, 2], [rnd(1, 4, 3, 5), rnd(4, 3, 3, 3), rnd(6)])
add("conv_t_depthwise", "conv_transpose2d", [2, 1, 1, 1, 3], [rnd(1, 3, 4, 4), rnd(3, 1, 3, 3), rnd(3)])
add("conv_t_k2s2", "conv_transpose2d_nobias", [2, 0, 0, 1, 1], [rnd(2, 6, 3, 4), rnd(6, 3, 2, 2)])
add("conv_t_big", "conv_transpose2d", [2, 1, 0, 1, 1], [rnd(4, 64, 7, 7), rnd(64, 32, 4, 4), rnd(32)])
# attrs: mode (0 nearest, 1 bilinear) align_corners scale / mode align_corners height width
add("up_nearest2", "upsample", [0, 0, 2.0], [rnd(2, 3, 4, 5)])
add("up_nearest3", "upsample", [0, 0, 3.0], [rnd(1, 2, 3, 2)])
add("up_bilinear2", "upsample", [1, 0, 2.0], [rnd(2, 3, 4, 5)])
add("up_bilinear_align", "upsample", [1, 1, 2.0], [rnd(2, 3, 4, 5)])
add("up_bilinear_1.5", "upsample", [1, 0, 1.5], [rnd(1, 2, 5, 6)])
add("resize_bilinear", "interpolate", [1, 0, 7, 9], [rnd(2, 3, 4, 5)])
add("resize_bilinear_down", "interpolate", [1, 0, 3, 2], [rnd(1, 2, 7, 5)])
add("resize_nearest", "interpolate", [0, 0, 3, 7], [rnd(2, 2, 5, 4)])
add("resize_align_1", "interpolate", [1, 1, 1, 6], [rnd(1, 2, 3, 4)])
# ---- activations (attrs: alpha / slope / threshold / min max)
for op, attrs in [("elu", [1.0]), ("elu_0.5", [0.5]), ("selu", []), ("celu", [1.3]), ("mish", []), ("softsign", []),
                  ("hardsigmoid", []), ("hardswish", []), ("hardtanh", [-0.7, 1.2]), ("thresholded_relu", [0.4]), ("cube", [])]:
    add(op, op.split("_0")[0], attrs, [rnd(13, 11, lo=-5, hi=5)])
add("elu_small", "elu", [1.0], [rnd(9, 7, lo=-1e-3, hi=1e-3)])
add("prelu_shared", "prelu", [], [rnd(4, 3, 5), rnd(1)])
add("prelu_channels", "prelu", [], [rnd(2, 6, 4, 3), rnd(6)])
add("rrelu_eval", "rrelu_eval", [0.125, 1 / 3], [rnd(9, 8)])
# ---- math (inputs inside each domain), comparisons, logic, where, one-hot
for op, gen in [("sin", rnd), ("cos", rnd), ("tan", lambda *d: rnd(*d, lo=-1.3, hi=1.3)), ("asin", lambda *d: rnd(*d, lo=-0.95, hi=0.95)),
                ("acos", lambda *d: rnd(*d, lo=-0.95, hi=0.95)), ("atan", lambda *d: rnd(*d, lo=-5, hi=5)), ("sinh", lambda *d: rnd(*d, lo=-4, hi=4)),
                ("cosh", lambda *d: rnd(*d, lo=-4, hi=4)), ("asinh", lambda *d: rnd(*d, lo=-5, hi=5)), ("acosh", lambda *d: rnd(*d, lo=1.1, hi=6)),
                ("atanh", lambda *d: rnd(*d, lo=-0.9, hi=0.9)), ("erf", lambda *d: rnd(*d, lo=-3, hi=3)), ("erfc", lambda *d: rnd(*d, lo=-3, hi=3)),
                ("log1p", lambda *d: rnd(*d, lo=-0.9, hi=5)), ("expm1", lambda *d: rnd(*d, lo=-3, hi=3))]:
    add(f"math_{op}", op, [], [gen(9, 7)])
add("math_sinh_small", "sinh", [], [rnd(5, 4, lo=-1e-3, hi=1e-3)])
add("math_erf_small", "erf", [], [rnd(5, 4, lo=-0.4, hi=0.4)])
add("math_expm1_small", "expm1", [], [rnd(5, 4, lo=-1e-4, hi=1e-4)])
halves = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 3.49, -3.51, 0.0, 7.0], np.float32)
for op in ("round", "trunc", "floor", "ceil", "sign"):
    add(f"math_{op}", op, [], [np.concatenate([rnd(30, lo=-5, hi=5), halves])], grads=[False])
add("atan2", "atan2", [], [rnd(7, 6), rnd(7, 6)])
add("atan2_broadcast", "atan2", [], [rnd(4, 5), rnd(5)])
ints = lambda *d: rng.integers(-2, 3, size=d).astype(np.float32)
for op in ("ne", "le", "logical_and", "logical_or", "logical_xor"):
    add(op, op, [], [ints(6, 7), ints(7)], grads=[False, False])
add("logical_not", "logical_not", [], [ints(6, 7)], grads=[False])
add("where", "where", [], [(rnd(6, 7) > 0).astype(np.float32), rnd(6, 7), rnd(7)], grads=[False, True, True])
add("one_hot", "one_hot", [5], [np.array([0, 4, 2, 2, 1, 3, 0], np.int32)])
# ---- losses (attrs: reduction 0 none, 1 mean, 2 sum, then the loss's parameter)
signs = lambda *d: np.where(rng.uniform(size=d) < 0.5, -1.0, 1.0).astype(np.float32)
add("l1_mean", "l1", [1], [rnd(6, 7), rnd(6, 7)])
add("l1_none", "l1", [0], [rnd(5, 3), rnd(5, 3)])
add("huber", "huber", [1, 1.0], [rnd(8, 9, lo=-3, hi=3), rnd(8, 9, lo=-3, hi=3)])
add("huber_0.3_sum", "huber", [2, 0.3], [rnd(6, 5), rnd(6, 5)])
add("smooth_l1", "smooth_l1", [1, 0.5], [rnd(8, 9), rnd(8, 9)])
add("nll_mean", "nll_mean", [], [rnd(9, 6, lo=-4, hi=-0.1), np.array([0, 5, 2, 3, 1, 4, 0, 2, 5], np.int32)])
add("hinge", "hinge", [1], [rnd(7, 5), signs(7, 5)])
add("squared_hinge", "squared_hinge", [1], [rnd(7, 5), signs(7, 5)])
add("cosine_similarity", "cosine_similarity", [1], [rnd(6, 9), rnd(6, 9)])
add("cosine_proximity", "cosine_proximity", [1], [rnd(6, 9), rnd(6, 9)])
add("poisson_log", "poisson_log", [1], [rnd(6, 5), pos(6, 5)])
add("poisson", "poisson", [1], [pos(6, 5), pos(6, 5)])
# ---- metric losses (attrs: reduction, then margin), CTC (attrs: reduction, blank, input lengths, target lengths), sparsemax
add("triplet_margin", "triplet_margin", [1, 1.0], [rnd(7, 6), rnd(7, 6), rnd(7, 6)])
add("contrastive", "contrastive", [1, 1.5], [rnd(8, 5, lo=-1, hi=1), rnd(8, 5, lo=-1, hi=1), (rng.uniform(size=8) < 0.5).astype(np.float32)], [True, True, False])
add("cosine_embedding", "cosine_embedding", [1, 0.2], [rnd(8, 6), rnd(8, 6), signs(8)], [True, True, False])
add("margin_ranking_sum", "margin_ranking", [2, 0.3], [rnd(9), rnd(9), signs(9)], [True, True, False])
add("ctc_mean", "ctc", [1, 0, 12, 9, 11, 4, 2, 3], [rnd(12, 3, 6, lo=-3, hi=3), np.array([[1, 2, 2, 5], [3, 3, 0, 0], [4, 1, 4, 0]], np.int32)])
add("ctc_none_blank3", "ctc", [0, 3, 20, 17, 7, 5], [rnd(20, 2, 5, lo=-3, hi=3), np.array([[1, 0, 1, 2, 4, 4, 2], [2, 2, 1, 0, 4, 0, 0]], np.int32)])
add("sparsemax", "sparsemax", [], [rnd(6, 9, lo=-3, hi=3)])
# ---- linear algebra (multi-part results flattened and concatenated; vectors signed as jitNN signs them)
def spd(*shape):
    m = rng.uniform(-1, 1, size=shape)
    return (m @ np.swapaxes(m, -1, -2) + shape[-1] * np.eye(shape[-1])).astype(np.float32)


def sym(*shape):
    m = rng.uniform(-2, 2, size=shape)
    return ((m + np.swapaxes(m, -1, -2)) / 2).astype(np.float32)


def tri(n):
    return (rng.uniform(-1, 1, size=(n, n)) + 3 * np.eye(n)).astype(np.float32)


add("lu_factor", "lu_factor", [], [rnd(5, 5)])
add("lu_factor_batch", "lu_factor", [], [rnd(3, 4, 4)])
add("solve_triangular_lower", "solve_triangular", [0, 0], [tri(5), rnd(5, 3)])
add("solve_triangular_upper", "solve_triangular", [1, 0], [tri(5), rnd(5, 3)])
add("solve_triangular_upper_unit", "solve_triangular", [1, 1], [tri(4), rnd(4, 2)])
add("solve", "solve", [], [rnd(6, 6), rnd(6, 2)])
add("solve_batch", "solve", [], [rnd(2, 4, 4), rnd(2, 4, 3)])
add("inv", "inv", [], [rnd(5, 5)])
add("det", "det", [], [rnd(5, 5)])
add("det_batch", "det", [], [rnd(3, 4, 4)])
add("slogdet", "slogdet", [], [rnd(3, 4, 4)])
add("cholesky", "cholesky", [], [spd(5, 5)])
add("cholesky_batch", "cholesky", [], [spd(2, 4, 4)])
add("qr_tall", "qr", [], [rnd(6, 4)])
add("qr_batch", "qr", [], [rnd(2, 5, 5)])
add("qr_wide", "qr", [], [rnd(3, 5)])
add("eigh", "eigh", [], [sym(5, 5)])
add("eigh_batch", "eigh", [], [sym(2, 6, 6)])
add("svd_tall", "svd", [], [rnd(6, 4)])
add("svd_wide", "svd", [], [rnd(4, 6)])
add("svd_batch", "svd", [], [rnd(2, 5, 5)])
# ---- Fourier transforms on (re, im) pairs (attrs: n, dim, norm 0 backward / 1 ortho / 2 forward); stft/istft (attrs: n_fft, hop, center, normalized, window 0 none / 1 hann)
add("fft", "fft", [-1, -1, 0], [rnd(3, 12, 2)])
add("fft_pad_ortho", "fft", [16, -1, 1], [rnd(2, 10, 2)])
add("fft_crop_dim0", "fft", [5, 0, 2], [rnd(6, 4, 2)])
add("ifft", "ifft", [-1, -1, 0], [rnd(2, 9, 2)])
add("rfft", "rfft", [-1, -1, 0], [rnd(3, 10)])
add("rfft_odd_ortho", "rfft", [-1, -1, 1], [rnd(2, 9)])
add("irfft", "irfft", [-1, -1, 0], [rnd(2, 6, 2)])
add("irfft_n9", "irfft", [9, -1, 0], [rnd(2, 5, 2)])
add("fft2", "fft2", [], [rnd(3, 4, 5, 2)])
add("rfft2", "rfft2", [], [rnd(2, 4, 6)])
add("stft_hann", "stft", [16, 4, 1, 0, 1], [rnd(2, 64)])
add("stft_rect_nocenter", "stft", [8, 3, 0, 1, 0], [rnd(30)])
add("istft_hann", "istft", [16, 4, 1, 0, 1], [rnd(2, 9, 17, 2)])
# ---- sparse times dense (attrs: rows, cols, nnz, row indices, column indices; duplicates add up) and GCN (attrs: nodes, edges, sources, destinations)
sp_r = [0, 2, 2, 4, 1, 0, 3, 4, 2, 1, 4]
sp_c = [1, 0, 5, 3, 3, 1, 2, 0, 4, 0, 5]
add("spmm", "spmm", [5, 6, len(sp_r)] + sp_r + sp_c, [rnd(len(sp_r)), rnd(6, 7)])
g_src = [0, 1, 1, 2, 3, 4, 5, 5, 0]
g_dst = [1, 0, 2, 3, 2, 5, 4, 0, 5]
add("gcn", "gcn", [6, len(g_src)] + g_src + g_dst, [rnd(6, 4), rnd(3, 4)])
# ---- data-dependent sizes (unique attrs: 0 values, 1 inverse, 2 counts)
def sparse(*shape):
    a = rnd(*shape)
    a[rng.uniform(size=shape) < 0.5] = 0
    return a


add("nonzero", "nonzero", [], [sparse(4, 5)], [False])
add("nonzero_3d", "nonzero", [], [sparse(3, 4, 300)], [False])
add("nonzero_static", "nonzero_static", [12, -1], [sparse(5, 6)], [False])
add("masked_select", "masked_select", [], [rnd(5, 6), (rng.uniform(size=(5, 6)) < 0.5).astype(np.float32)], [True, False])
add("masked_select_broadcast", "masked_select", [], [rnd(700, 6), np.array([1, 0, 1, 1, 0, 0], np.float32)], [True, False])
dup = rng.integers(0, 7, (40, 30)).astype(np.float32)
add("unique", "unique", [0], [dup], [False])
add("unique_inverse", "unique", [1], [dup], [False])
add("unique_counts", "unique", [2], [dup], [False])
# ---- FP8 with a power-of-two scale (attrs: format 0 E4M3 / 1 E5M2, log2 of the scale)
add("fp8_roundtrip_e4m3", "fp8_roundtrip", [0, -6], [rnd(9, 13)], [False])
add("fp8_roundtrip_e5m2", "fp8_roundtrip", [1, -6], [rnd(9, 13, lo=-100, hi=100)], [False])
add("fp8_fake_quantize", "fp8_fake", [0, -5], [rnd(7, 11)])
add("fp8_linear_e4m3", "fp8_linear", [0, -6], [rnd(5, 40), rnd(9, 40), rnd(9)], [True, False, True])
add("fp8_linear_e5m2", "fp8_linear", [1, -4], [rnd(3, 2, 17), rnd(6, 17), rnd(6)], [True, False, True])
add("fp8_scaled_mm", "fp8_scaled_mm", [0, -6], [rnd(18, 33), rnd(21, 33)], [False, False])
# ---- GGUF block formats: random valid blocks (bytes as int32 0..255), dequantised by llama.cpp's gguf package
# (attrs: format 0 Q4_0, 1 Q4_1, 2 Q4_K, 3 Q5_K, 4 Q6_K; output rows N)
GGUF = [("Q4_0", 32, 18, [0]), ("Q4_1", 32, 20, [0, 2]), ("Q4_K", 256, 144, [0, 2]), ("Q5_K", 256, 176, [0, 2]), ("Q6_K", 256, 210, [208])]


def gguf_blocks(fmt, n, k):
    _, qk, nb, scales = GGUF[fmt]
    raw = rng.integers(0, 256, size=(n * k // qk, nb), dtype=np.uint8)
    for off in scales:
        raw[:, off:off + 2] = rng.uniform(0.002, 0.03, size=(raw.shape[0], 1)).astype(np.float16).view(np.uint8)
    return raw.reshape(-1).astype(np.int32)


for fmt in range(5):
    name = GGUF[fmt][0].lower()
    for rows, n in [(1, 23), (5, 24), (40, 20)]:
        add(f"gguf_linear_{name}_r{rows}", "gguf_linear", [fmt, n], [rnd(rows, 512), gguf_blocks(fmt, n, 512), rnd(n)], [True, False, True])
    add(f"gguf_dequant_{name}", "gguf_dequant", [fmt, 3], [gguf_blocks(fmt, 3, 512)], [False])
# ---- embedding-model losses
add("negative_sampling", "negative_sampling", [], [rnd(6, 5, lo=-1, hi=1), rnd(6, 5, lo=-1, hi=1), rnd(6, 4, 5, lo=-1, hi=1)])
add("glove_loss", "glove_loss", [], [rnd(7, 5, lo=-1, hi=1), rnd(7, 5, lo=-1, hi=1), rnd(7), rnd(7), pos(7), pos(7)], [True, True, True, True, False, False])
add("msle", "msle", [1], [pos(6, 5), pos(6, 5)])
add("mape", "mape", [1], [rnd(6, 5), rnd(6, 5, lo=0.5, hi=3) * signs(6, 5)])
# ---- norms (attrs: groups eps / eps / size alpha beta k)
add("group_norm", "group_norm", [3, 1e-5], [rnd(2, 6, 4, 5), rnd(6, lo=0.5, hi=1.5), rnd(6)])
add("group_norm_1", "group_norm", [1, 1e-5], [rnd(3, 4, 7), rnd(4, lo=0.5, hi=1.5), rnd(4)])
add("instance_norm", "instance_norm", [1e-5], [rnd(2, 3, 6, 5)])
add("lrn", "lrn", [3, 1e-2, 0.75, 1.0], [rnd(2, 7, 4, 3)])
add("lrn_even", "lrn", [4, 0.5, 0.6, 2.0], [rnd(1, 9, 5)])
add("weight_norm", "weight_norm", [], [rnd(5, 3, 2, 2), rnd(5, lo=0.5, hi=2)])
# ---- 1-D and 3-D convolution and pooling, adaptive pooling, padding
add("conv1d", "conv1d", [2, 1, 1], [rnd(2, 3, 11), rnd(5, 3, 3), rnd(5)])
add("conv1d_dil", "conv1d", [1, 2, 2], [rnd(1, 4, 9), rnd(3, 4, 3), rnd(3)])
add("conv3d", "conv3d", [1, 1, 1], [rnd(1, 2, 4, 5, 3), rnd(3, 2, 3, 3, 3), rnd(3)])
add("conv3d_s2", "conv3d", [2, 0, 1], [rnd(2, 2, 5, 4, 6), rnd(4, 2, 2, 2, 2), rnd(4)])
for op, dims in [("max_pool1d", (2, 3, 11)), ("avg_pool1d", (2, 3, 11)), ("max_pool3d", (1, 2, 5, 4, 6)), ("avg_pool3d", (1, 2, 5, 4, 6))]:
    add(op, op, [3, 2, 1], [rnd(*dims)])
add("max_pool1d_k2", "max_pool1d", [2, 2, 0], [rnd(2, 2, 8)])
for op, dims, out in [("adaptive_avg", (2, 3, 10), [4]), ("adaptive_avg", (1, 2, 7, 9), [3, 4]), ("adaptive_avg", (1, 2, 5, 6, 7), [2, 3, 3]),
                      ("adaptive_max", (2, 3, 10), [4]), ("adaptive_max", (1, 2, 7, 9), [3, 4]), ("adaptive_avg", (1, 3, 6, 6), [1, 1])]:
    add(f"{op}_{len(out)}d_{'x'.join(map(str, out))}", op, out, [rnd(*dims)])
for mode in (0, 1, 2):
    add(f"pad1d_m{mode}", "pad", [mode, 2, 3], [rnd(2, 3, 6)])
    add(f"pad2d_m{mode}", "pad", [mode, 1, 2, 2, 1], [rnd(1, 2, 4, 5)])
add("pad3d_reflect", "pad", [1, 1, 1, 2, 0, 1, 1], [rnd(1, 1, 3, 4, 3)])
# ---- scans and sorting (attrs: axis / axis descending)
add("cumsum_0", "cumsum", [0], [rnd(7, 5)])
add("cumsum_last", "cumsum", [-1], [rnd(3, 4, 9)])
add("cumprod_1", "cumprod", [1], [rnd(4, 6, 3, lo=0.5, hi=1.5)])
with_zeros = rnd(3, 7, lo=-1.5, hi=1.5)
with_zeros[0, 2] = 0.0
with_zeros[1, 5] = 0.0
with_zeros[2, 1] = 0.0
with_zeros[2, 4] = 0.0
add("cumprod_zeros", "cumprod", [1], [with_zeros])
ties = rng.integers(-3, 4, size=(4, 11)).astype(np.float32)
add("sort_asc", "sort", [1, 0], [rnd(5, 13)])
add("sort_desc_axis0", "sort", [0, 1], [rnd(9, 4)])
add("sort_ties", "sort", [1, 0], [ties])
add("argsort_ties", "argsort", [1, 0], [ties])
add("argsort_desc", "argsort", [-1, 1], [ties])
# ---- indexing (attrs: axis / axis / mode offsets...)
add("index_select_0", "index_select", [0], [rnd(6, 5), np.array([4, 0, 4, 2], np.int32)])
add("index_select_1", "index_select", [1], [rnd(3, 7, 2), np.array([6, 1, 1, 0, 3], np.int32)])
add("gather_1", "gather", [1], [rnd(4, 6), rng.integers(0, 6, size=(4, 3)).astype(np.int32)])
add("gather_0", "gather", [0], [rnd(5, 3, 2), rng.integers(0, 5, size=(7, 3, 2)).astype(np.int32)])
add("scatter_add", "scatter_add", [1], [rnd(4, 6), rng.integers(0, 6, size=(4, 9)).astype(np.int32), rnd(4, 9)])
uniq = np.stack([rng.permutation(6)[:4] for _ in range(3)]).astype(np.int32)
add("scatter_unique", "scatter", [1], [rnd(3, 6), uniq, rnd(3, 4)], grads=[False, False, False])
for mode in (0, 1, 2):
    add(f"embedding_bag_m{mode}", "embedding_bag", [mode, 0, 3, 3, 7], [np.array([1, 4, 2, 0, 0, 3, 9, 4, 4, 2], np.int32), rnd(10, 5)])
# ---- masked and bidirectional recurrent layers (attrs: bidirectional, lengths...)
def rnn_weights(gates, i, h):
    w = lambda *d: rnd(*d, lo=-0.5, hi=0.5)
    return [w(gates * h, i), w(gates * h, h), w(gates * h), w(gates * h)]
add("lstm_masked", "rnn_lstm", [0, 5, 3, 1], [rnd(5, 3, 4, lo=-1, hi=1)] + rnn_weights(4, 4, 6) + [rnd(1, 3, 6, lo=-1, hi=1), rnd(1, 3, 6, lo=-1, hi=1)])
add("lstm_bidirectional", "rnn_lstm", [1, 4, 4], [rnd(4, 2, 3, lo=-1, hi=1)] + rnn_weights(4, 3, 5) + rnn_weights(4, 3, 5))
add("gru_bidirectional_masked", "rnn_gru", [1, 6, 2, 4], [rnd(6, 3, 4, lo=-1, hi=1)] + rnn_weights(3, 4, 5) + rnn_weights(3, 4, 5))
# ---- image operations
add("flip_hw", "flip", [2, 3], [rnd(2, 3, 4, 5)])
add("flip_c", "flip", [1], [rnd(2, 3, 4, 5)])
add("center_crop", "center_crop", [3, 4], [rnd(2, 3, 6, 7)])
add("normalize_image", "normalize_image", [], [rnd(2, 3, 4, 5, lo=0, hi=1)])
add("rgb_to_grayscale", "rgb_to_grayscale", [], [rnd(2, 3, 4, 5, lo=0, hi=1)])
# ---- NaN-aware operators (NaN and infinities injected)
def specials(*d):
    a = rnd(*d)
    flat = a.reshape(-1)
    flat[::5] = np.nan
    flat[1::7] = np.inf
    flat[3::11] = -np.inf
    return a
nan_x, nan_y = specials(6, 7), specials(6, 7)
for op in ("isnan", "isinf", "isfinite"):
    add(op, op, [], [nan_x], grads=[False])
add("nan_to_num", "nan_to_num", [0.5, 7.0], [nan_x], grads=[False])
add("nan_to_num_grad", "nan_to_num", [0.5, 7.0], [rnd(4, 5)])
for op in ("maximum_nan", "minimum_nan", "fmax", "fmin"):
    add(op, op, [], [nan_x, nan_y], grads=[False, False])
    add(f"{op}_plain", op, [], [rnd(5, 6), rnd(6)])
# ---- recurrent layers over [T, B, I]; inputs x, W_ih, W_hh, b_ih, b_hh, h0 (, c0)
def rnn_case(name, op, t, b, i, h, gates):
    w = lambda *s: rnd(*s, lo=-0.5, hi=0.5)
    states = [rnd(b, h, lo=-1, hi=1)] + ([rnd(b, h, lo=-1, hi=1)] if op == "lstm" else [])
    add(name, op, [], [rnd(t, b, i, lo=-1, hi=1), w(gates * h, i), w(gates * h, h), w(gates * h), w(gates * h)] + states)


rnn_case("lstm", "lstm", 5, 3, 4, 6, 4)
rnn_case("lstm_t1", "lstm", 1, 2, 3, 5, 4)
rnn_case("lstm_big", "lstm", 9, 4, 16, 33, 4)
rnn_case("gru", "gru", 5, 3, 4, 6, 3)
rnn_case("gru_t1", "gru", 1, 2, 3, 5, 3)
rnn_case("gru_big", "gru", 9, 4, 16, 33, 3)


def rounded(x, kind):
    return (x.half() if kind == 0 else x.bfloat16()).double()


class HalfOperands(torch.autograd.Function):
    """Rounds a tensor to half precision; its gradient is rounded too, as jitNN casts the
    incoming gradient of a half-precision GEMM to the operands' type."""

    @staticmethod
    def forward(ctx, x, kind):
        ctx.kind = kind
        return rounded(x, kind)

    @staticmethod
    def backward(ctx, g):
        return g, None


class HalfGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, kind):
        ctx.kind = kind
        return y.clone()

    @staticmethod
    def backward(ctx, g):
        return rounded(g, ctx.kind), None


def forward(op, attrs, t):
    if op in ("add", "sub", "mul", "div"):
        return getattr(torch, op)(t[0], t[1])
    if op == "maximum":
        return torch.maximum(t[0], t[1])
    if op in ("neg", "exp", "log", "sqrt", "abs", "relu", "reciprocal", "square"):
        return getattr(torch, op)(t[0])
    if op == "pow":
        return torch.pow(t[0], attrs[0])
    if op == "clamp":
        return torch.clamp(t[0], attrs[0], attrs[1])
    if op == "affine":
        return t[0] * attrs[0] + attrs[1]
    if op in ("sum", "mean"):
        return getattr(torch, op)(t[0], dim=attrs[0], keepdim=bool(attrs[1]))
    if op in ("max", "min"):
        return getattr(torch, "amax" if op == "max" else "amin")(t[0], dim=attrs[0], keepdim=bool(attrs[1]))
    if op == "sum_all":
        return t[0].sum()
    if op == "mean_all":
        return t[0].mean()
    if op == "argmax":
        return torch.argmax(t[0], dim=attrs[0])
    if op == "matmul":
        a = t[0].transpose(-1, -2) if attrs[0] else t[0]
        b = t[1].transpose(-1, -2) if attrs[1] else t[1]
        return a @ b
    if op == "mha_mask":
        B, T, D = t[0].shape
        H = attrs[0]
        q, k, v = (t[0] @ t[1].T + t[2]).reshape(B, T, 3, H, D // H).permute(2, 0, 3, 1, 4)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=t[5])
        return y.permute(0, 2, 1, 3).reshape(B, T, D) @ t[3].T + t[4]
    if op in ("sdpa_bthd", "sdpa_bthd_mask"):
        scale = None if attrs[1] < 0 else attrs[1]
        mask = t[3] if op == "sdpa_bthd_mask" else None
        q, k, v = (x.permute(0, 2, 1, 3) for x in t[:3])
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=bool(attrs[0]), scale=scale)
        return y.permute(0, 2, 1, 3)
    if op == "rope_heads":
        return forward("rope", attrs, [t[0].permute(0, 2, 1, 3)]).permute(0, 2, 1, 3)
    if op == "mha":
        T = t[0].shape[1]
        mask = torch.ones(T, T, dtype=torch.bool).triu(1) if attrs[1] else None
        y, _ = torch.nn.functional.multi_head_attention_forward(
            t[0].transpose(0, 1), t[0].transpose(0, 1), t[0].transpose(0, 1), t[0].shape[2], attrs[0], t[1], t[2],
            None, None, False, 0.0, t[3], t[4], training=False, need_weights=False, attn_mask=mask, is_causal=bool(attrs[1]))
        return y.transpose(0, 1)
    if op == "cast_f16":
        return t[0].half().double()
    if op == "cast_bf16":
        return t[0].bfloat16().double()
    if op == "matmul_half":
        kind = attrs[2]
        a = HalfOperands.apply(t[0], kind)
        b = HalfOperands.apply(t[1], kind)
        a = a.transpose(-1, -2) if attrs[0] else a
        b = b.transpose(-1, -2) if attrs[1] else b
        return HalfGrad.apply(a @ b, kind)
    if op == "linear_half":
        kind = attrs[0]
        y = HalfGrad.apply(HalfOperands.apply(t[0], kind) @ HalfOperands.apply(t[1], kind).T, kind)
        return y + t[2]
    if op == "topk":
        return torch.topk(t[0], attrs[0], dim=-1).values
    if op == "rope":
        base, offset, inter = attrs
        T, D = t[0].shape[-2], t[0].shape[-1]
        pos = torch.arange(offset, offset + T, dtype=torch.float64)
        freq = torch.tensor([float(base) ** (-2.0 * i / D) for i in range(D // 2)], dtype=torch.float64)
        ang = torch.outer(pos, freq)
        c, s = torch.cos(ang), torch.sin(ang)
        x = t[0]
        if inter:
            a, b = x[..., 0::2], x[..., 1::2]
            return torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)
        a, b = x[..., : D // 2], x[..., D // 2:]
        return torch.cat([a * c - b * s, a * s + b * c], dim=-1)
    if op in ("sdpa", "sdpa_mask"):
        scale = None if attrs[1] < 0 else attrs[1]
        mask = t[3] if op == "sdpa_mask" else None
        if attrs[0] and t[0].shape[-2] != t[1].shape[-2]:
            # PyTorch's is_causal mask is tril(ones(T, S)), aligned to the top left
            T, S = t[0].shape[-2], t[1].shape[-2]
            keep = torch.ones(T, S, dtype=torch.bool).tril()
            mask = torch.zeros(T, S).masked_fill(~keep, float("-inf"))
            return torch.nn.functional.scaled_dot_product_attention(t[0], t[1], t[2], attn_mask=mask, scale=scale)
        return torch.nn.functional.scaled_dot_product_attention(t[0], t[1], t[2], attn_mask=mask, is_causal=bool(attrs[0]), scale=scale)
    if op == "permute":
        return t[0].permute(*attrs)
    if op == "reshape":
        return t[0].reshape(*attrs)
    if op == "softmax":
        return torch.softmax(t[0], dim=-1)
    if op == "log_softmax":
        return torch.log_softmax(t[0], dim=-1)
    if op == "cross_entropy":
        return torch.nn.functional.cross_entropy(t[0], t[1].long())
    if op == "linear":
        return torch.nn.functional.linear(t[0], t[1], t[2])
    if op == "linear_relu":
        return torch.relu(torch.nn.functional.linear(t[0], t[1], t[2]))
    if op == "linear_nobias":
        return torch.nn.functional.linear(t[0], t[1])
    if op == "conv2d":
        return torch.nn.functional.conv2d(t[0], t[1], t[2], stride=attrs[0], padding=attrs[1], dilation=attrs[2], groups=attrs[3])
    if op == "conv2d_nobias":
        return torch.nn.functional.conv2d(t[0], t[1], None, stride=attrs[0], padding=attrs[1], dilation=attrs[2], groups=attrs[3])
    if op == "batch_norm":
        c = t[0].shape[1]
        rm, rv = torch.zeros(c), torch.ones(c)
        y = torch.nn.functional.batch_norm(t[0], rm, rv, t[1], t[2], training=True, momentum=0.1, eps=1e-5)
        extras["rm"], extras["rv"] = rm, rv
        return y
    if op == "max_pool2d":
        return torch.nn.functional.max_pool2d(t[0], attrs[0], attrs[1], attrs[2])
    if op == "avg_pool2d":
        return torch.nn.functional.avg_pool2d(t[0], attrs[0], attrs[1], attrs[2])
    if op == "global_avg_pool":
        return torch.nn.functional.adaptive_avg_pool2d(t[0], 1).flatten(1)
    if op == "relu6":
        return torch.nn.functional.relu6(t[0])
    if op in ("tanh", "sigmoid"):
        return getattr(torch, op)(t[0])
    if op == "gelu":
        return torch.nn.functional.gelu(t[0])
    if op == "gelu_tanh":
        return torch.nn.functional.gelu(t[0], approximate="tanh")
    if op == "silu":
        return torch.nn.functional.silu(t[0])
    if op == "swiglu":
        return torch.nn.functional.silu(t[0]) * t[1]
    if op == "embedding":
        return torch.nn.functional.embedding(t[0].long(), t[1], padding_idx=None if attrs[0] < 0 else attrs[0])
    if op == "layer_norm":
        return torch.nn.functional.layer_norm(t[0], (t[0].shape[-1],), t[1], t[2], eps=attrs[0])
    if op == "layer_norm_nobias":
        return torch.nn.functional.layer_norm(t[0], (t[0].shape[-1],), t[1], None, eps=attrs[0])
    if op == "rms_norm":
        return torch.nn.functional.rms_norm(t[0], (t[0].shape[-1],), t[1], eps=attrs[0])
    if op == "narrow":
        return t[0].narrow(attrs[0], attrs[1], attrs[2])
    if op == "concat":
        return torch.cat(t, dim=attrs[0])
    if op == "stack":
        return torch.stack(t, dim=attrs[0])
    if op == "leaky_relu":
        return torch.nn.functional.leaky_relu(t[0], attrs[0])
    if op == "softplus":
        return torch.nn.functional.softplus(t[0])
    if op == "mse":
        return torch.nn.functional.mse_loss(t[0], t[1])
    if op == "mse_sum":
        return torch.nn.functional.mse_loss(t[0], t[1], reduction="sum")
    if op == "bce":
        return torch.nn.functional.binary_cross_entropy(t[0], t[1])
    if op == "bce_logits":
        return torch.nn.functional.binary_cross_entropy_with_logits(t[0], t[1])
    if op == "gaussian_kl":
        return -0.5 * torch.sum(1 + t[1] - t[0] ** 2 - t[1].exp())
    if op in ("conv_transpose2d", "conv_transpose2d_nobias"):
        bias = t[2] if op == "conv_transpose2d" else None
        return torch.nn.functional.conv_transpose2d(t[0], t[1], bias, stride=attrs[0], padding=attrs[1], output_padding=attrs[2], dilation=attrs[3], groups=attrs[4])
    if op == "upsample":
        mode = "nearest" if attrs[0] == 0 else "bilinear"
        return torch.nn.functional.interpolate(t[0], scale_factor=attrs[2], mode=mode, align_corners=None if attrs[0] == 0 else bool(attrs[1]))
    if op == "interpolate":
        mode = "nearest" if attrs[0] == 0 else "bilinear"
        return torch.nn.functional.interpolate(t[0], size=(attrs[2], attrs[3]), mode=mode, align_corners=None if attrs[0] == 0 else bool(attrs[1]))
    if op == "lstm":
        return torch._VF.lstm(t[0], (t[5].unsqueeze(0), t[6].unsqueeze(0)), t[1:5], True, 1, 0.0, True, False, False)[0]
    if op == "gru":
        return torch._VF.gru(t[0], t[5].unsqueeze(0), t[1:5], True, 1, 0.0, True, False, False)[0]
    if op in ("elu", "celu"):
        return getattr(torch.nn.functional, op)(t[0], attrs[0])
    if op in ("selu", "mish", "softsign", "hardsigmoid", "hardswish"):
        return getattr(torch.nn.functional, op)(t[0])
    if op == "hardtanh":
        return torch.nn.functional.hardtanh(t[0], attrs[0], attrs[1])
    if op == "thresholded_relu":
        return torch.where(t[0] > attrs[0], t[0], torch.zeros_like(t[0]))
    if op == "cube":
        return t[0] ** 3
    if op == "prelu":
        return torch.nn.functional.prelu(t[0], t[1])
    if op == "rrelu_eval":
        return torch.nn.functional.rrelu(t[0], attrs[0], attrs[1], training=False)
    if op in ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "erf", "erfc", "log1p", "expm1", "round", "trunc", "floor", "ceil", "sign"):
        return getattr(torch, op)(t[0])
    if op == "atan2":
        return torch.atan2(t[0], t[1])
    if op in ("ne", "le"):
        return getattr(torch, op)(t[0], t[1]).to(t[0].dtype)
    if op in ("logical_and", "logical_or", "logical_xor"):
        return getattr(torch, op)(t[0], t[1]).to(t[0].dtype)
    if op == "logical_not":
        return torch.logical_not(t[0]).to(t[0].dtype)
    if op == "where":
        return torch.where(t[0] != 0, t[1], t[2])
    if op == "one_hot":
        return torch.nn.functional.one_hot(t[0].long(), attrs[0]).to(torch.float64)
    if op in ("l1", "huber", "smooth_l1", "hinge", "squared_hinge", "poisson_log", "poisson", "msle", "mape", "cosine_proximity"):
        red = ["none", "mean", "sum"][attrs[0]]
        x, y = t[0], t[1]
        if op == "l1":
            return torch.nn.functional.l1_loss(x, y, reduction=red)
        if op == "huber":
            return torch.nn.functional.huber_loss(x, y, reduction=red, delta=attrs[1])
        if op == "smooth_l1":
            return torch.nn.functional.smooth_l1_loss(x, y, reduction=red, beta=attrs[1])
        if op == "poisson_log":
            return torch.nn.functional.poisson_nll_loss(x, y, log_input=True, reduction=red)
        if op == "poisson":
            return torch.nn.functional.poisson_nll_loss(x, y, log_input=False, eps=1e-8, reduction=red)
        l = {"hinge": lambda: torch.relu(1 - y * x), "squared_hinge": lambda: torch.relu(1 - y * x) ** 2,
             "msle": lambda: (torch.log1p(x) - torch.log1p(y)) ** 2,
             "mape": lambda: 100 * (y - x).abs() / y.abs().clamp_min(1e-7),
             "cosine_proximity": lambda: -torch.nn.functional.cosine_similarity(x, y, dim=-1, eps=1e-8)}[op]()
        return l.mean() if red == "mean" else (l.sum() if red == "sum" else l)
    if op in ("triplet_margin", "contrastive", "cosine_embedding", "margin_ranking"):
        red = ["none", "mean", "sum"][attrs[0]]
        F = torch.nn.functional
        if op == "triplet_margin":
            return F.triplet_margin_loss(t[0], t[1], t[2], margin=attrs[1], reduction=red)
        if op == "cosine_embedding":
            return F.cosine_embedding_loss(t[0], t[1], t[2], margin=attrs[1], reduction=red)
        if op == "margin_ranking":
            return F.margin_ranking_loss(t[0], t[1], t[2], margin=attrs[1], reduction=red)
        d = F.pairwise_distance(t[0], t[1])
        l = t[2] * d ** 2 + (1 - t[2]) * torch.relu(attrs[1] - d) ** 2
        return l.mean() if red == "mean" else (l.sum() if red == "sum" else l)
    if op == "ctc":
        b = t[0].shape[1]
        lens = attrs[2:]
        return torch.nn.functional.ctc_loss(torch.log_softmax(t[0], -1), t[1].long(), torch.tensor(lens[:b]), torch.tensor(lens[b:]), blank=attrs[1],
                                            reduction=["none", "mean", "sum"][attrs[0]])
    if op == "sparsemax":
        z = torch.sort(t[0], -1, descending=True)[0]
        k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
        support = (1 + k * z > z.cumsum(-1)).to(z.dtype)
        tau = ((support * z).sum(-1, keepdim=True) - 1) / support.sum(-1, keepdim=True)
        return torch.relu(t[0] - tau)
    if op in ("lu_factor", "solve_triangular", "solve", "inv", "det", "slogdet", "cholesky", "qr", "eigh", "svd"):
        L = torch.linalg
        flat = lambda *xs: torch.cat([x.reshape(-1) for x in xs])
        signs_of = lambda x: torch.sign(torch.gather(x, -2, x.abs().argmax(-2, keepdim=True))).detach()
        if op == "lu_factor":
            return L.lu_factor(t[0])[0]
        if op == "solve_triangular":
            return L.solve_triangular(t[0], t[1], upper=bool(attrs[0]), unitriangular=bool(attrs[1]))
        if op == "solve":
            return L.solve(t[0], t[1])
        if op == "inv":
            return L.inv(t[0])
        if op == "det":
            return L.det(t[0])
        if op == "slogdet":
            return L.slogdet(t[0]).logabsdet
        if op == "cholesky":
            return L.cholesky(t[0])
        if op == "qr":
            return flat(*L.qr(t[0]))
        if op == "eigh":
            w, v = L.eigh(t[0])
            return flat(w, v * signs_of(v))
        u, sv, vh = L.svd(t[0], full_matrices=False)
        sg = signs_of(u)
        return flat(u * sg, sv, vh.mT * sg)
    if op in ("fft", "ifft", "rfft", "irfft", "fft2", "rfft2", "stft", "istft"):
        fft = torch.fft
        cx = lambda x: torch.view_as_complex(x.contiguous())
        rx = torch.view_as_real
        if op in ("fft", "ifft", "rfft", "irfft"):
            norm = ["backward", "ortho", "forward"][attrs[2]]
            n = None if attrs[0] < 0 else attrs[0]
            x = t[0] if op == "rfft" else cx(t[0])
            y = getattr(fft, op)(x, n=n, dim=attrs[1], norm=norm)
            return y if op == "irfft" else rx(y)
        if op == "fft2":
            return rx(fft.fft2(cx(t[0])))
        if op == "rfft2":
            return rx(fft.rfft2(t[0]))
        n_fft, hop, center, normalized, win = attrs
        window = torch.hann_window(n_fft, dtype=torch.float64) if win else None
        if op == "stft":
            return rx(torch.stft(t[0], n_fft, hop, window=window, center=bool(center), normalized=bool(normalized), onesided=True, return_complex=True))
        return torch.istft(cx(t[0]), n_fft, hop, window=window, center=bool(center), normalized=bool(normalized), onesided=True)
    if op == "spmm":
        rows, cols, nnz = attrs[:3]
        a = torch.zeros(rows, cols).index_put((torch.tensor(attrs[3:3 + nnz]), torch.tensor(attrs[3 + nnz:])), t[0], accumulate=True)
        return a @ t[1]
    if op == "gcn":
        n, e = attrs[:2]
        src, dst = attrs[2:2 + e], attrs[2 + e:]
        a = torch.zeros(n, n)
        for s_, d_ in zip(src, dst):
            a[d_, s_] += 1
        a = a + torch.eye(n)
        deg = a.sum(1)
        norm = a / torch.sqrt(deg[:, None] * deg[None, :])
        return norm @ (t[0] @ t[1].T)
    if op == "negative_sampling":
        ls = torch.nn.functional.logsigmoid
        h, tg, ng = t
        return -(ls((h * tg).sum(-1)) + ls(-(ng * h[:, None, :]).sum(-1)).sum(-1)).mean()
    if op == "glove_loss":
        wi, wj, bi, bj, lx, w = t
        return (w * ((wi * wj).sum(-1) + bi + bj - lx) ** 2).mean()
    if op == "nonzero":
        return torch.nonzero(t[0])
    if op == "nonzero_static":
        return torch.nonzero_static(t[0], size=attrs[0], fill_value=attrs[1])
    if op == "masked_select":
        return torch.masked_select(t[0], t[1] != 0)
    if op == "unique":
        v, inv, cnt = torch.unique(t[0], sorted=True, return_inverse=True, return_counts=True)
        return [v, inv, cnt][attrs[0]]
    if op in ("fp8_roundtrip", "fp8_fake", "fp8_linear", "fp8_scaled_mm"):
        f8 = [torch.float8_e4m3fn, torch.float8_e5m2][attrs[0]]
        sc = 2.0 ** attrs[1]
        rt = lambda v: (v.detach().float() / sc).to(f8).to(torch.float64) * sc
        if op == "fp8_roundtrip":
            return rt(t[0])
        if op == "fp8_fake":
            return t[0] + (rt(t[0]) - t[0]).detach()
        if op == "fp8_linear":
            return t[0] @ rt(t[1]).T + t[2]
        return rt(t[0]) @ rt(t[1]).T
    if op in ("gguf_linear", "gguf_dequant"):
        import gguf
        qt = getattr(gguf.GGMLQuantizationType, GGUF[attrs[0]][0])
        raw = t[1] if op == "gguf_linear" else t[0]
        n = attrs[1]
        w = torch.from_numpy(gguf.quants.dequantize(raw.numpy().astype(np.uint8).reshape(n, -1), qt).astype(np.float64))
        if op == "gguf_dequant":
            return w
        return t[0] @ w.T + t[2]
    if op == "nll_mean":
        return torch.nn.functional.nll_loss(t[0], t[1].long())
    if op == "cosine_similarity":
        return torch.nn.functional.cosine_similarity(t[0], t[1], dim=attrs[0], eps=1e-8)
    if op == "group_norm":
        return torch.nn.functional.group_norm(t[0], attrs[0], t[1], t[2], eps=attrs[1])
    if op == "instance_norm":
        return torch.nn.functional.instance_norm(t[0], eps=attrs[0])
    if op == "lrn":
        return torch.nn.functional.local_response_norm(t[0], attrs[0], alpha=attrs[1], beta=attrs[2], k=attrs[3])
    if op == "weight_norm":
        return torch._weight_norm(t[0], t[1].reshape(-1, 1, 1, 1), 0)
    if op == "conv1d":
        return torch.nn.functional.conv1d(t[0], t[1], t[2], stride=attrs[0], padding=attrs[1], dilation=attrs[2])
    if op == "conv3d":
        return torch.nn.functional.conv3d(t[0], t[1], t[2], stride=attrs[0], padding=attrs[1], dilation=attrs[2])
    if op in ("max_pool1d", "avg_pool1d", "max_pool3d", "avg_pool3d"):
        return getattr(torch.nn.functional, op)(t[0], attrs[0], attrs[1], attrs[2])
    if op == "adaptive_avg":
        return getattr(torch.nn.functional, f"adaptive_avg_pool{len(attrs)}d")(t[0], attrs)
    if op == "adaptive_max":
        return getattr(torch.nn.functional, f"adaptive_max_pool{len(attrs)}d")(t[0], attrs)
    if op == "pad":
        mode = ["constant", "reflect", "replicate"][attrs[0]]
        sides = attrs[1:]
        r = len(sides) // 2
        torch_pad = []
        for k in reversed(range(r)):
            torch_pad += [sides[2 * k], sides[2 * k + 1]]
        return torch.nn.functional.pad(t[0], torch_pad, mode=mode, value=0.5 if mode == "constant" else None) if mode == "constant" else torch.nn.functional.pad(t[0], torch_pad, mode=mode)
    if op in ("cumsum", "cumprod"):
        return getattr(torch, op)(t[0], dim=attrs[0])
    if op == "sort":
        return torch.sort(t[0], dim=attrs[0], descending=bool(attrs[1]), stable=True).values
    if op == "argsort":
        return torch.sort(t[0], dim=attrs[0], descending=bool(attrs[1]), stable=True).indices
    if op == "index_select":
        return torch.index_select(t[0], attrs[0], t[1].long())
    if op == "gather":
        return torch.gather(t[0], attrs[0], t[1].long())
    if op == "scatter_add":
        return torch.scatter_add(t[0], attrs[0], t[1].long(), t[2])
    if op == "scatter":
        return torch.scatter(t[0], attrs[0], t[1].long(), t[2])
    if op == "embedding_bag":
        return torch.nn.functional.embedding_bag(t[0].long(), t[1], torch.tensor(attrs[1:]), mode=["sum", "mean", "max"][attrs[0]])
    if op in ("rnn_lstm", "rnn_gru"):
        bidir, lengths = bool(attrs[0]), attrs[1:]
        cls = torch.nn.LSTM if op == "rnn_lstm" else torch.nn.GRU
        x = t[0]
        h = t[2].shape[1]
        net = cls(x.shape[2], h, bidirectional=bidir).double()
        names = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]
        if bidir:
            names += [n + "_reverse" for n in names]
        params = dict(zip(names, t[1:1 + len(names)]))
        states = t[1 + len(names):]
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, torch.tensor(lengths), enforce_sorted=False)
        hx = (states[0], states[1]) if op == "rnn_lstm" and states else None
        out, _ = torch.func.functional_call(net, params, (packed, hx))
        return torch.nn.utils.rnn.pad_packed_sequence(out, total_length=x.shape[0])[0]
    if op == "flip":
        return torch.flip(t[0], attrs)
    if op == "center_crop":
        h, w = t[0].shape[-2:]
        top, left = int(round((h - attrs[0]) / 2.0)), int(round((w - attrs[1]) / 2.0))
        return t[0][..., top:top + attrs[0], left:left + attrs[1]]
    if op == "normalize_image":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).double().reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).double().reshape(1, 3, 1, 1)
        return (t[0] - mean) * (1 / std)
    if op == "rgb_to_grayscale":
        w = torch.tensor([0.2989, 0.587, 0.114], dtype=torch.float32).double().reshape(1, 3, 1, 1)
        return (t[0] * w).sum(1, keepdim=True)
    if op in ("isnan", "isinf", "isfinite"):
        return getattr(torch, op)(t[0]).to(torch.float64)
    if op == "nan_to_num":
        return torch.nan_to_num(t[0], nan=attrs[0], posinf=attrs[1], neginf=-attrs[1])
    if op in ("maximum_nan", "minimum_nan"):
        return getattr(torch, op.split("_")[0])(t[0], t[1])
    if op in ("fmax", "fmin"):
        return getattr(torch, op)(t[0], t[1])
    raise ValueError(op)


arrays = []  # (case, role, np array)
extras = {}
lines = []

for name, op, attrs, inputs, grads in cases:
    ts = [torch.tensor(a.astype(np.float64), requires_grad=g) if a.dtype == np.float32 else torch.tensor(a) for a, g in zip(inputs, grads)]
    extras.clear()
    y = forward(op, attrs, ts)
    for role, v in extras.items():
        arrays.append((name, role, v.detach().numpy().astype(np.float32)))
    lines.append(f"case {name} {op} " + " ".join(str(a) for a in attrs))
    for i, a in enumerate(inputs):
        arrays.append((name, f"in{i}", a))
    if y.dtype == torch.int64:
        arrays.append((name, "out", y.numpy().astype(np.int32)))
        continue
    w = rnd(*y.shape) if y.dim() > 0 else rnd(1).reshape(())
    if y.requires_grad:
        (y * torch.tensor(w.astype(np.float64))).sum().backward()
    arrays.append((name, "w", w))
    arrays.append((name, "out", y.detach().numpy().astype(np.float32)))
    for i, tt in enumerate(ts):
        if grads[i]:
            arrays.append((name, f"grad{i}", tt.grad.numpy().astype(np.float32)))

# ---- optimisers: three steps from a fixed start with fixed gradients
for name, make in [
    ("sgd", lambda p: torch.optim.SGD([p], lr=0.1, weight_decay=0.01)),
    ("sgd_momentum", lambda p: torch.optim.SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)),
    ("sgd_nesterov", lambda p: torch.optim.SGD([p], lr=0.1, momentum=0.9, nesterov=True)),
    ("adam", lambda p: torch.optim.Adam([p], lr=0.01, weight_decay=0.01)),
    ("adamw", lambda p: torch.optim.AdamW([p], lr=0.01, weight_decay=0.05)),
    ("amsgrad", lambda p: torch.optim.Adam([p], lr=0.01, weight_decay=0.01, amsgrad=True)),
    ("adagrad", lambda p: torch.optim.Adagrad([p], lr=0.1, lr_decay=0.01, weight_decay=0.01, initial_accumulator_value=0.1)),
    ("rmsprop", lambda p: torch.optim.RMSprop([p], lr=0.01, alpha=0.9, weight_decay=0.01)),
    ("rmsprop_centered_momentum", lambda p: torch.optim.RMSprop([p], lr=0.01, alpha=0.9, momentum=0.8, centered=True)),
    ("adadelta", lambda p: torch.optim.Adadelta([p], lr=1.0, rho=0.9, eps=1e-6, weight_decay=0.01)),
    ("adamax", lambda p: torch.optim.Adamax([p], lr=0.02, weight_decay=0.01)),
    ("nadam", lambda p: torch.optim.NAdam([p], lr=0.01, weight_decay=0.01)),
]:
    p0 = rnd(37)
    p = torch.tensor(p0.astype(np.float64), requires_grad=True)
    opt = make(p)
    lines.append(f"case opt_{name} optim {name}")
    arrays.append((f"opt_{name}", "p0", p0))
    for k in range(3):
        g = rnd(37)
        arrays.append((f"opt_{name}", f"g{k}", g))
        p.grad = torch.tensor(g.astype(np.float64))
        opt.step()
    arrays.append((f"opt_{name}", "out", p.detach().numpy().astype(np.float32)))

# ---- layer-wise optimisers, not in torch.optim: their update rules in FP64
class Lamb:
    def __init__(self, p, lr, b1, b2, eps, wd):
        self.p, self.lr, self.b1, self.b2, self.eps, self.wd, self.t = p, lr, b1, b2, eps, wd, 0
        self.m, self.v = torch.zeros_like(p), torch.zeros_like(p)

    def step(self):
        with torch.no_grad():
            self.t += 1
            g = self.p.grad
            self.m = self.b1 * self.m + (1 - self.b1) * g
            self.v = self.b2 * self.v + (1 - self.b2) * g * g
            u = (self.m / (1 - self.b1 ** self.t)) / ((self.v / (1 - self.b2 ** self.t)).sqrt() + self.eps) + self.wd * self.p
            wn, un = self.p.norm(), u.norm()
            r = wn / un if wn > 0 and un > 0 else 1.0
            self.p -= self.lr * r * u


class Lars:
    def __init__(self, p, lr, momentum, wd, eta, eps):
        self.p, self.lr, self.mu, self.wd, self.eta, self.eps = p, lr, momentum, wd, eta, eps
        self.buf = torch.zeros_like(p)

    def step(self):
        with torch.no_grad():
            g = self.p.grad
            wn, gn = self.p.norm(), g.norm()
            local = self.eta * wn / (gn + self.wd * wn + self.eps) if wn > 0 and gn > 0 else 1.0
            self.buf = self.mu * self.buf + self.lr * local * (g + self.wd * self.p)
            self.p -= self.buf


for name, make in [("lamb", lambda p: Lamb(p, 0.01, 0.9, 0.999, 1e-6, 0.01)), ("lars", lambda p: Lars(p, 0.1, 0.9, 1e-4, 0.001, 1e-9))]:
    p0 = rnd(37)
    p = torch.tensor(p0.astype(np.float64), requires_grad=True)
    opt = make(p)
    lines.append(f"case opt_{name} optim {name}")
    arrays.append((f"opt_{name}", "p0", p0))
    for k in range(3):
        g = rnd(37)
        arrays.append((f"opt_{name}", f"g{k}", g))
        p.grad = torch.tensor(g.astype(np.float64))
        opt.step()
    arrays.append((f"opt_{name}", "out", p.detach().numpy().astype(np.float32)))

# ---- jitNN's Philox init, reproduced independently in numpy
lines.append("case init_uniform uniform 42 -1.0 3.0")
arrays.append(("init_uniform", "out", uniform(42, -1.0, 3.0, 4099)))

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
offset = 0
with open(os.path.join(out, "data.bin"), "wb") as f:
    for case, role, a in arrays:
        dt = "f32" if a.dtype == np.float32 else "i32"
        dims = " ".join(str(d) for d in a.shape)
        lines.append(f"{case} {role} {dt} {offset} {dims}".rstrip())
        b = np.ascontiguousarray(a).astype("<f4" if dt == "f32" else "<i4").tobytes()
        f.write(b)
        offset += len(b)
with open(os.path.join(out, "manifest.txt"), "w") as f:
    f.write(f"# generated by scripts/pytorch/goldens.py with torch {torch.__version__}\n")
    f.write("\n".join(lines) + "\n")
print(f"{sum(1 for l in lines if l.startswith('case '))} cases, {len(arrays)} arrays, {offset} bytes")
