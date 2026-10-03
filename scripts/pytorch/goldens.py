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
# ---- wave 2: convolution (attrs: stride pad dilation groups), batch norm, pooling, relu6
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
# ---- wave 3: activations
for op in ["tanh", "sigmoid", "gelu", "gelu_tanh", "silu"]:
    add(op, op, [], [rnd(17, 13, lo=-6, hi=6)])
    add(f"{op}_tails", op, [], [rnd(64, lo=-12, hi=12)])
add("swiglu", "swiglu", [], [rnd(9, 14, lo=-4, hi=4), rnd(9, 14)])
# ---- wave 3: slicing and joining (attrs: axis start length / axis)
add("narrow_mid", "narrow", [1, 2, 4], [rnd(4, 9, 5)])
add("narrow_first", "narrow", [0, 1, 2], [rnd(3, 7)])
add("narrow_last", "narrow", [2, 5, 3], [rnd(2, 3, 8)])
add("narrow_size1", "narrow", [1, 0, 1], [rnd(1, 6, 1)])
add("concat_mid", "concat", [1], [rnd(2, 3, 4), rnd(2, 1, 4), rnd(2, 5, 4)])
add("concat_first", "concat", [0], [rnd(3, 6), rnd(4, 6)])
add("concat_last", "concat", [-1], [rnd(5, 2), rnd(5, 7)])
add("stack_0", "stack", [0], [rnd(3, 4), rnd(3, 4), rnd(3, 4)])
add("stack_last", "stack", [2], [rnd(3, 4), rnd(3, 4)])
# ---- wave 3: embedding (repeated, unused and padding rows); attrs: padding index or -1
add("embedding", "embedding", [-1], [rng.integers(0, 11, (3, 5)).astype(np.int32), rnd(11, 6)])
add("embedding_padding", "embedding", [2], [np.array([2, 0, 2, 5, 5, 5, 1], np.int32), rnd(7, 4)])
add("embedding_wide", "embedding", [-1], [rng.integers(0, 50, 300).astype(np.int32), rnd(50, 33)])
# ---- wave 3: batched matmul (equal batches, or one side shared) and attention
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
pad = np.zeros((2, 1, 1, 7), np.float32)
pad[0, 0, 0, 5:] = -1e9
pad[1, 0, 0, 3:] = -1e9
add("sdpa_padding", "sdpa_mask", [0, -1], [rnd(2, 3, 5, 8), rnd(2, 3, 7, 8), rnd(2, 3, 7, 6), pad], [True, True, True, False])
# multi-head self-attention (attrs: heads causal); inputs x, Wqkv, bqkv, Wout, bout
for name, causal in [("mha", 0), ("mha_causal", 1)]:
    add(name, "mha", [4, causal], [rnd(2, 6, 16), rnd(48, 16, lo=-0.5, hi=0.5), rnd(48), rnd(16, 16, lo=-0.5, hi=0.5), rnd(16)])
# rotary position embedding (attrs: base offset interleaved), computed directly in FP64
add("rope_half", "rope", [10000.0, 0, 0], [rnd(2, 3, 5, 8)])
add("rope_interleaved", "rope", [10000.0, 0, 1], [rnd(2, 3, 5, 8)])
add("rope_offset", "rope", [500000.0, 7, 0], [rnd(1, 2, 4, 16)])
add("rope_long", "rope", [10000.0, 0, 0], [rnd(1, 1, 300, 64)])
# top-k values (attrs: k); random inputs have no ties
add("topk", "topk", [3], [rnd(5, 11)])
add("topk_long", "topk", [8], [rnd(2, 3, 900)])
# ---- wave 3: LayerNorm and RMSNorm over the last axis; attrs: eps
add("layer_norm", "layer_norm", [1e-5], [rnd(7, 33), rnd(33, lo=0.5, hi=1.5), rnd(33)])
add("layer_norm_3d", "layer_norm", [1e-5], [rnd(2, 5, 48), rnd(48, lo=0.5, hi=1.5), rnd(48)])
add("layer_norm_long", "layer_norm", [1e-5], [rnd(3, 1000, lo=1, hi=5), rnd(1000, lo=0.5, hi=1.5), rnd(1000)])
add("layer_norm_rows", "layer_norm", [1e-5], [rnd(150, 16), rnd(16, lo=0.5, hi=1.5), rnd(16)])
add("layer_norm_nobias", "layer_norm_nobias", [1e-5], [rnd(6, 20), rnd(20, lo=0.5, hi=1.5)])
add("rms_norm", "rms_norm", [1e-6], [rnd(7, 33), rnd(33, lo=0.5, hi=1.5)])
add("rms_norm_3d", "rms_norm", [1e-6], [rnd(2, 5, 48), rnd(48, lo=0.5, hi=1.5)])
add("rms_norm_long", "rms_norm", [1e-6], [rnd(3, 1000), rnd(1000, lo=0.5, hi=1.5)])
add("rms_norm_rows", "rms_norm", [1e-6], [rnd(150, 16), rnd(16, lo=0.5, hi=1.5)])


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
    if op == "mha":
        T = t[0].shape[1]
        mask = torch.ones(T, T, dtype=torch.bool).triu(1) if attrs[1] else None
        y, _ = torch.nn.functional.multi_head_attention_forward(
            t[0].transpose(0, 1), t[0].transpose(0, 1), t[0].transpose(0, 1), t[0].shape[2], attrs[0], t[1], t[2],
            None, None, False, 0.0, t[3], t[4], training=False, need_weights=False, attn_mask=mask, is_causal=bool(attrs[1]))
        return y.transpose(0, 1)
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
