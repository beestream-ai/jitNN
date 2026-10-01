"""Step-0 gradients of the CNN gate models in FP64 (the "true" gradient), and how far PyTorch's
own FP32 gradients are from it per parameter. Writes <prefix>.grad0_f64 (float64 values, in
parameters() order) and <prefix>.grad0_f32err (float32 relative L2 error of PyTorch FP32, per
parameter).

usage: python cnn_grad64.py <resnet18|mobilenetv2> <data dir> <prefix> [mnist|cifar10]
"""
import sys

import numpy as np
import os

import torch

# A true FP32 reference: no TF32 in cuDNN convolutions or matmuls on NVIDIA GPUs.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from datasets import load
from models import mobilenet_v2, resnet18


def grads(kind, xtr, ytr, dtype):
    c = xtr.shape[1]
    net = resnet18(c, 10, True, 1) if kind == "resnet18" else mobilenet_v2(c, 10, True, 1)
    dev = torch.device(os.environ.get("JITNN_TORCH_DEVICE", "cpu"))
    net = net.to(dtype).to(dev)
    net.train()
    x = torch.from_numpy(xtr[:64]).to(dtype).to(dev)
    y = torch.from_numpy(ytr[:64]).to(dev)
    torch.nn.functional.cross_entropy(net(x), y).backward()
    return [p.grad.detach().double().cpu().numpy().ravel() for p in net.parameters()]


def main():
    kind, d, prefix = sys.argv[1], sys.argv[2], sys.argv[3]
    xtr, ytr, _, _, _ = load(sys.argv[4] if len(sys.argv) > 4 else "mnist", d)
    g64 = grads(kind, xtr, ytr, torch.float64)
    g32 = grads(kind, xtr, ytr, torch.float32)
    err = [np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30) for a, b in zip(g64, g32)]
    np.concatenate(g64).astype(np.float64).tofile(prefix + ".grad0_f64")
    np.array(err, dtype=np.float32).tofile(prefix + ".grad0_f32err")
    print(f"{kind}: {len(err)} parameters, PyTorch FP32 vs FP64 relative error: max {max(err):.2e}, median {np.median(err):.2e}")


if __name__ == "__main__":
    main()
