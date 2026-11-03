"""Training step time of PyTorch on jitNN's training workloads (jitnn-bench Workloads.java, same
models, seeds, optimisers and losses): one variant, one workload, one precision per call.

usage: python gtrain.py <t1 dir> <work dir> <variant> <workload> <precision>
  variant: eager | fused (fused optimiser, channels-last CNNs) | compile | compile-ma (max-autotune-no-cudagraphs) | compile-ro (reduce-overhead)
  workload: resnet50_cifar_train_b32 | resnet18_train_b64 | gpt2_train_b8_s256 | bert_train_b8_s128
  precision: fp32 (TF32 off) | tf32 | fp16 (autocast to FP16)
Protocol: 10 warm-up steps, then the median of 30 steps, each ending with loss.item()."""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

t1dir, work, variant, wl, prec = sys.argv[1:6]
sys.path.insert(0, t1dir)
import t1  # noqa: E402

models, transformers = t1.models, t1.transformers
WARMUP, STEPS = 10, 30
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = prec == "tf32"
torch.backends.cudnn.allow_tf32 = prec == "tf32"
torch.backends.cudnn.benchmark = True
fused = variant != "eager"
torch.manual_seed(0)
g = torch.Generator().manual_seed(0)
cnn = wl.startswith("resnet")
if wl == "resnet50_cifar_train_b32":
    net, b, classes = models.resnet50(3, 10, True, 1), 32, 10
elif wl == "resnet18_train_b64":
    net, b, classes = models.resnet18(3, 10, True, 1), 64, 10
elif wl == "gpt2_train_b8_s256":
    net, b, t, vocab = transformers.GPT2(50257, 1024, 768, 12, 12, 1), 8, 256, 50257
else:
    net, b, t, vocab = t1.BertEncoder(), 8, 128, 30522
net = net.to(dev).train()
if cnn:
    x = (torch.rand(b, 3, 32, 32, generator=g) * 2 - 1).to(dev)
    y = (torch.arange(b) % classes).to(dev)
    if fused:
        net = net.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, fused=fused or None)
else:
    x = torch.from_numpy(t1.tokens(b, t) % vocab).to(dev)
    y = torch.from_numpy(((torch.arange(b * t) * 7919 + 13) % vocab).numpy()).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4 if wl.startswith("gpt2") else 1e-4, fused=fused or None)


def loss_fn(net, x, y):
    with torch.autocast("cuda", dtype=torch.float16, enabled=prec == "fp16"):
        out = net(x)
        if cnn:
            return F.cross_entropy(out, y)
        if wl.startswith("gpt2"):
            return F.cross_entropy(out.reshape(b * t, -1).float(), y)
        return out.float().square().mean()


fwd = loss_fn
if variant.startswith("compile"):
    mode = {"compile": None, "compile-ma": "max-autotune-no-cudagraphs", "compile-ro": "reduce-overhead"}[variant]
    fwd = torch.compile(loss_fn, mode=mode)


def step():
    loss = fwd(net, x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


t0 = time.perf_counter()
first = step()
setup = time.perf_counter() - t0
for _ in range(WARMUP - 1):
    step()
ms = []
for _ in range(STEPS):
    t0 = time.perf_counter()
    step()
    ms.append((time.perf_counter() - t0) * 1000)
ms.sort()
med, p10, p90 = ms[STEPS // 2], ms[STEPS // 10], ms[STEPS * 9 // 10]
row = dict(fw="pytorch", variant=variant, model=wl, prec=prec, step_ms=med, p10=p10, p90=p90, first_loss=first, setup_s=setup,
           torch_mem_mb=torch.cuda.max_memory_allocated() / 2**20, torch=torch.__version__)
with open(os.path.join(work, "gtrain.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[gtrain] pytorch {variant:10s} {wl:26s} {prec:4s} {med:9.2f} ms/step (p10 {p10:8.2f} p90 {p90:8.2f})  loss0 {first:.4f}  setup {setup:6.1f} s", flush=True)
