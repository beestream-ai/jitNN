"""PyTorch baselines for the kernel benchmark: for each call in a jitNN results file
(jitnn-bench, one JSON line per call), the same operation and shapes through PyTorch on the same GPU,
which launches the vendor kernels (cuBLAS, cuDNN, flash/memory-efficient SDPA). TF32 off for FP32.

Covered: matmul, linear, conv2d and its two gradients, flash attention forward and backward,
LSTM and GRU recurrences forward and backward (cuDNN layers, which also project their input),
softmax, row norms (as LayerNorm), batch norm, cross entropy, embedding, element-wise unary and
binary (as neg and add, which move the same bytes). Other operators get no baseline; their
efficiency against the bandwidth and compute ceilings still applies.

Protocol: 20 warm-up calls, then 100 timed calls with CUDA events, median, p10, p90.

usage: python kernels.py <jitnn results.jsonl> <baselines.jsonl>
"""
import json
import sys

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
DEV = "cuda"


def tensor(spec, positive=False):
    dt = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}.get(spec["dtype"])
    if dt is None:
        return torch.zeros(spec["shape"], dtype=torch.long, device=DEV)
    t = torch.rand(spec["shape"], dtype=dt, device=DEV) * 2 - 1
    return t.abs() + 0.1 if positive else t


def timed(fn, warmup=20, runs=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1000)
    ts.sort()
    return ts[runs // 2], ts[runs // 10], ts[runs * 9 // 10]


def build(r):
    """A zero-argument callable doing the row's operation in PyTorch, or None."""
    op, a, ins = r["op"], r["iattr"], r["in"]
    if op == "matmul":
        x, y = tensor(ins[0]), tensor(ins[1])
        m, n, k, ta, tb = a[:5]
        if len(a) > 5:
            batch = a[5]
            x = x.reshape(batch, k, m) if ta else x.reshape(batch, m, k)
            y = y.reshape(batch, n, k) if tb else y.reshape(batch, k, n)
        xa = x.transpose(-1, -2) if ta else x
        yb = y.transpose(-1, -2) if tb else y
        return lambda: torch.matmul(xa, yb)
    if op == "linear":
        x, w = tensor(ins[0]), tensor(ins[1])
        b = tensor(ins[2]) if len(ins) > 2 else None
        relu = len(a) > 4 and a[4] == 1
        return (lambda: torch.relu(F.linear(x, w, b))) if relu else (lambda: F.linear(x, w, b))
    if op in ("conv2d", "conv2d_bwd_data", "conv2d_bwd_weight"):
        n, cin, h, w_, cout, oh, ow, k, stride, pad, dil, groups = a[:12]
        if op == "conv2d":
            x, wt = tensor(ins[0]), tensor(ins[1])
            b = tensor(ins[2]) if len(ins) > 2 and ins[2]["shape"] == [cout] else None
            return lambda: F.conv2d(x, wt, b, stride, pad, dil, groups)
        if op == "conv2d_bwd_data":
            dy, wt = tensor(ins[0]), tensor(ins[1])
            return lambda: torch.nn.grad.conv2d_input([n, cin, h, w_], wt, dy, stride, pad, dil, groups)
        dy, x = tensor(ins[0]), tensor(ins[1])
        return lambda: torch.nn.grad.conv2d_weight(x, [cout, cin // groups, k, k], dy, stride, pad, dil, groups)
    if op in ("flash_attention", "flash_attention_bwd"):
        z, t, s, d, causal = a[:5]
        q = torch.rand(1, z, t, d, device=DEV)
        kk = torch.rand(1, z, s, d, device=DEV)
        v = torch.rand(1, z, s, d, device=DEV)
        if op == "flash_attention":
            return lambda: F.scaled_dot_product_attention(q, kk, v, is_causal=bool(causal))
        q.requires_grad_(); kk.requires_grad_(); v.requires_grad_()
        o = F.scaled_dot_product_attention(q, kk, v, is_causal=bool(causal))
        go = torch.rand_like(o)
        return lambda: torch.autograd.grad(o, (q, kk, v), go, retain_graph=True)
    if op in ("lstm", "gru", "lstm_bwd", "gru_bwd"):
        # jitNN's op is the recurrence only (input gates precomputed by a GEMM); cuDNN's layer also
        # projects its input, so this baseline does a little more work than the jitNN call
        t, b, h = a[:3]
        cell = torch.nn.LSTM(h, h) if op.startswith("lstm") else torch.nn.GRU(h, h)
        cell = cell.to(DEV)
        x = torch.rand(t, b, h, device=DEV, requires_grad=op.endswith("_bwd"))
        if not op.endswith("_bwd"):
            return lambda: cell(x)
        y = cell(x)[0]
        gy = torch.rand_like(y)
        return lambda: torch.autograd.grad(y, [x] + list(cell.parameters()), gy, retain_graph=True)
    if op == "softmax":
        x = tensor(ins[0])
        return lambda: torch.softmax(x, -1)
    if op == "row_norm":
        x = tensor(ins[0])
        d = x.shape[-1]
        w = torch.ones(d, device=DEV)
        return lambda: F.layer_norm(x, (d,), w, w)
    if op == "batch_norm":
        x = tensor(ins[0])
        c = x.shape[1]
        w = torch.ones(c, device=DEV)
        return lambda: F.batch_norm(x, None, None, w, w, training=True)
    if op == "cross_entropy":
        x = tensor(ins[0])
        y = torch.zeros(x.shape[0], dtype=torch.long, device=DEV)
        return lambda: F.cross_entropy(x, y)
    if op == "embedding":
        ids = torch.zeros(ins[0]["shape"], dtype=torch.long, device=DEV)
        w = tensor(ins[1])
        return lambda: F.embedding(ids, w)
    if op == "unary":
        x = tensor(ins[0])
        return lambda: torch.neg(x)
    if op == "binary":
        x, y = tensor(ins[0]), tensor(ins[1])
        return lambda: torch.add(x, y)
    return None


def main(src, dst):
    seen = set()
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            r = json.loads(line)
            key = json.dumps([r["op"], r["iattr"], r["fattr"], r["in"], r["out"]])
            if key in seen:
                continue
            seen.add(key)
            try:
                fn = build(r)
            except Exception as e:  # noqa: BLE001 - a shape PyTorch rejects gets no baseline
                print(f"[baseline] {r['op']}: {e}", flush=True)
                fn = None
            if fn is None:
                continue
            med, p10, p90 = timed(fn)
            out.write(json.dumps(dict(key=key, op=r["op"], torch_us=med, torch_us_p10=p10, torch_us_p90=p90, torch=torch.__version__)) + "\n")
            out.flush()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
