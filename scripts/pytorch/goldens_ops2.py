"""Golden tensors from PyTorch for the operators that close the gap to the other frameworks (remainder,
fmod, product, argmin, norms, Lp normalisation, depth/space rearrangements, tile, eye, arange, fold,
bitwise operations, GatherND and ScatterND). The same format as goldens.py, with its own random
stream, written to jitnn-core/src/test/resources/golden/ops2/ so the existing goldens stay unchanged.

usage: python goldens_ops2.py <output dir>
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

rng = np.random.default_rng(4321)
torch.set_default_dtype(torch.float64)

cases = []


def rnd(*shape, lo=-2.0, hi=2.0):
    return rng.uniform(lo, hi, size=shape).astype(np.float32)


def away(*shape):
    """Values with |v| in [0.5, 2], either sign: divisors and factors away from zero."""
    return (rng.uniform(0.5, 2.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)).astype(np.float32)


def ints(lo, hi, *shape):
    return rng.integers(lo, hi, size=shape).astype(np.int32)


def add(name, op, attrs, inputs, grads=None):
    cases.append((name, op, attrs, inputs, grads if grads is not None else [a.dtype == np.float32 for a in inputs]))


# ---- remainder and fmod (divisor away from zero, either sign; with broadcasting)
for op in ["remainder", "fmod"]:
    add(f"{op}_same", op, [], [rnd(7, 9, lo=-6, hi=6), away(7, 9)])
    add(f"{op}_row", op, [], [rnd(5, 11, lo=-6, hi=6), away(11)])
# ---- product, argmin, norms
add("prod_ax1", "prod", [1, 0], [away(4, 5, 3)])
add("prod_ax0_keep", "prod", [0, 1], [away(6, 4)])
add("argmin_ax1", "argmin", [1], [rnd(13, 29)], [False])
add("argmin_ax0", "argmin", [0], [rnd(13, 29)], [False])
add("norm2_ax1_keep", "norm", [2, 1, 1], [rnd(9, 14)])
add("norm1_ax0", "norm", [1, 0, 0], [away(9, 14)])
add("lp_normalize_2", "lp_normalize", [2, 1, 0.0], [rnd(6, 10, 3)])
add("lp_normalize_1_eps", "lp_normalize", [1, -1, 1e-12], [away(5, 8)])
# ---- rearrangements
add("depth_to_space_dcr", "depth_to_space", [2, 0], [rnd(2, 12, 3, 4)])
add("depth_to_space_crd", "depth_to_space", [2, 1], [rnd(2, 12, 3, 4)])
add("space_to_depth", "space_to_depth", [2], [rnd(2, 3, 6, 4)])
add("tile_3d", "tile", [2, 1, 3], [rnd(2, 3)])
add("tile_2d", "tile", [3, 2], [rnd(4, 1)])
add("fold_2d", "fold", [7, 6, 3, 2, 2, 1, 1, 1, 1, 1], [rnd(2, 3 * 3 * 2, 4 * 7)])
add("fold_stride2", "fold", [8, 8, 3, 3, 2, 2, 1, 1, 1, 1], [rnd(2, 2 * 9, 4 * 4)])
# ---- creation
add("eye_like_k1", "eye_like", [1], [rnd(3, 5)], [False])
add("eye_like_kneg", "eye_like", [-2], [rnd(4, 4)], [False])
add("arange", "arange", [-1.5, 3.0, 0.5], [], [])
# ---- bitwise on INT32
a32, b32 = ints(-1000, 1000, 6, 7), ints(-1000, 1000, 7)
for op in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
    add(op, op, [], [a32, b32], [False, False])
add("bitwise_not", "bitwise_not", [], [a32], [False])
add("shift_left", "shift_left", [], [ints(-500, 500, 5, 6), ints(0, 8, 5, 6)], [False, False])
add("shift_right", "shift_right", [], [ints(-50000, 50000, 5, 6), ints(0, 8, 6)], [False, False])
# ---- GatherND, ScatterND
add("gather_nd_2", "gather_nd", [], [rnd(4, 5, 3), np.stack([ints(-4, 4, 6), ints(0, 5, 6)], axis=1)], [True, False])
add("gather_nd_3", "gather_nd", [], [rnd(3, 4, 2), np.stack([ints(0, 3, 2, 3), ints(0, 4, 2, 3), ints(-2, 2, 2, 3)], axis=2)], [True, False])
perm = rng.permutation(5 * 4)[:7]
add("scatter_nd_none", "scatter_nd", [0], [rnd(5, 4, 3), np.stack([perm // 4, perm % 4], axis=1).astype(np.int32), rnd(7, 3)])
add("scatter_nd_add", "scatter_nd", [1], [rnd(5, 4, 3), np.stack([ints(0, 5, 9), ints(-4, 4, 9)], axis=1), rnd(9, 3)])
add("scatter_nd_mul", "scatter_nd", [2], [rnd(5, 4), ints(0, 5, 6, 1), away(6, 4)], [False, False, False])


def nd_tuple(idx):
    return tuple(idx[..., t].long() for t in range(idx.shape[-1]))


def forward(op, attrs, t):
    if op == "remainder":
        return torch.remainder(t[0], t[1])
    if op == "fmod":
        return torch.fmod(t[0], t[1])
    if op == "prod":
        return torch.prod(t[0], dim=attrs[0], keepdim=bool(attrs[1]))
    if op == "argmin":
        return torch.argmin(t[0], dim=attrs[0])
    if op == "norm":
        return torch.linalg.vector_norm(t[0], ord=attrs[0], dim=attrs[1], keepdim=bool(attrs[2]))
    if op == "lp_normalize":
        return F.normalize(t[0], p=attrs[0], dim=attrs[1], eps=attrs[2])
    if op == "depth_to_space":
        x, b = t[0], attrs[0]
        n, c, h, w = x.shape
        if attrs[1] == 1:
            return F.pixel_shuffle(x, b)
        # ONNX DepthToSpace, DCR: reshape [n, b, b, c/b^2, h, w], transpose (0, 3, 4, 1, 5, 2)
        return x.reshape(n, b, b, c // (b * b), h, w).permute(0, 3, 4, 1, 5, 2).reshape(n, c // (b * b), h * b, w * b)
    if op == "space_to_depth":
        x, b = t[0], attrs[0]
        n, c, h, w = x.shape
        return x.reshape(n, c, h // b, b, w // b, b).permute(0, 3, 5, 1, 2, 4).reshape(n, c * b * b, h // b, w // b)
    if op == "tile":
        return t[0].repeat(*attrs)
    if op == "fold":
        return F.fold(t[0], output_size=(attrs[0], attrs[1]), kernel_size=(attrs[2], attrs[3]), stride=(attrs[4], attrs[5]),
                      padding=(attrs[6], attrs[7]), dilation=(attrs[8], attrs[9]))
    if op == "eye_like":
        n, m = t[0].shape
        return torch.tensor(np.eye(n, m, attrs[0]))
    if op == "arange":
        return torch.arange(attrs[0], attrs[1], attrs[2])
    if op in ("bitwise_and", "bitwise_or", "bitwise_xor"):
        return getattr(torch, op)(t[0], t[1]).long()
    if op == "bitwise_not":
        return torch.bitwise_not(t[0]).long()
    if op == "shift_left":
        return torch.bitwise_left_shift(t[0], t[1]).long()
    if op == "shift_right":
        return torch.bitwise_right_shift(t[0], t[1]).long()
    if op == "gather_nd":
        return t[0][nd_tuple(t[1])]
    if op == "scatter_nd":
        data, idx, upd = t
        if attrs[0] == 2:
            out = data.clone()
            for r in range(idx.shape[0]):
                out[tuple(idx[r].long())] = out[tuple(idx[r].long())] * upd[r]
            return out
        return data.index_put(nd_tuple(idx), upd, accumulate=attrs[0] == 1)
    raise ValueError(op)


lines, arrays = [], []
for name, op, attrs, inputs, grads in cases:
    ts = [torch.tensor(a.astype(np.float64), requires_grad=g) if a.dtype == np.float32 else torch.tensor(a) for a, g in zip(inputs, grads)]
    y = forward(op, attrs, ts)
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
    f.write(f"# generated by scripts/pytorch/goldens_ops2.py with torch {torch.__version__}\n")
    f.write("\n".join(lines) + "\n")
print(f"{sum(1 for l in lines if l.startswith('case '))} cases, {len(arrays)} arrays, {offset} bytes")
