"""Training step time of PyTorch eager on the models jitNN's training benchmarks use, built from the
same seeds: ResNet-50 (224 x 224, batch 32) and ResNet-18 (CIFAR form, 32 x 32, batch 128) with SGD
and momentum 0.9, and a GPT-2 decoder (byte vocabulary) with AdamW and gradient clipping at 1, in
FP32 and under BF16 autocast.

Protocol, as DataParallelBenchAccelTest and TransformerBenchAccelTest: random inputs, 5 warm-up
steps, then the median of 30 steps, each ending with loss.item(). TF32 off, unless --cudnn-tf32 lets
cuDNN's convolutions use it as PyTorch does by default (jitNN's cuDNN convolutions do too, unless
NVIDIA_TF32_OVERRIDE=0); cudnn.benchmark on (the tuning PyTorch recommends).

usage: python train_bench.py [--device cuda] [--cudnn-tf32] [--dim 768] [--layers 12] [--heads 12] [--seq 128] [--batch 16]
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pytorch"))
import models  # noqa: E402
import transformers  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True

WARMUP, STEPS = 5, 30


def median_step(step, dev):
    times = []
    for s in range(WARMUP + STEPS):
        t0 = time.perf_counter()
        step()
        if s >= WARMUP:
            times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def cnn(name, net, size, batch, classes, dev):
    net = net.to(dev).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(batch, 3, size, size, generator=g) - 0.5).to(dev)
    y = torch.randint(0, classes, (batch,), generator=g).to(dev)

    def step():
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    ms = median_step(step, dev)
    print(f"[train] pytorch {name:28s} batch {batch:4d}: {ms:8.2f} ms/step  {batch * 1000 / ms:9.1f} samples/s", flush=True)


def gpt2(args, dev, dtype):
    net = transformers.GPT2(256, args.seq, args.dim, args.heads, args.layers, 1).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(7)
    x = torch.randint(0, 256, (args.batch, args.seq), generator=g).to(dev)
    y = torch.randint(0, 256, (args.batch * args.seq,), generator=g).to(dev)

    def step():
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            loss = F.cross_entropy(net(x).reshape(args.batch * args.seq, -1), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.item()

    ms = median_step(step, dev)
    name = f"GPT-2 {args.dim}x{args.layers}" + (" BF16" if dtype is not None else "")
    tokens = args.batch * args.seq
    print(f"[train] pytorch {name:28s} {args.batch} x {args.seq}: {ms:8.2f} ms/step  {tokens * 1000 / ms:9.1f} tokens/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cudnn-tf32", action="store_true", help="cuDNN convolutions in TF32 (PyTorch's default)")
    ap.add_argument("--only", default="", help="cnn or gpt2 to run one family")
    args = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = args.cudnn_tf32
    dev = torch.device(args.device)
    print(f"[train] torch {torch.__version__}, {torch.cuda.get_device_name() if dev.type == 'cuda' else dev}, cuDNN TF32 {'on' if args.cudnn_tf32 else 'off'}", flush=True)
    if args.only in ("", "cnn"):
        cnn("ResNet-50 224x224", models.resnet50(3, 1000, False, 7), 224, 32, 1000, dev)
        cnn("ResNet-18 32x32", models.resnet18(3, 10, True, 7), 32, 128, 10, dev)
    if args.only in ("", "gpt2"):
        gpt2(args, dev, None)
        gpt2(args, dev, torch.bfloat16)


if __name__ == "__main__":
    main()
