"""PyTorch reference for jitNN's CNN gates: ResNet-18 or MobileNetV2 (small variants) on MNIST or CIFAR-10,
from jitNN's initial weights, Adam(1e-3), batch 64, the first <steps> training batches in order.
Writes [test accuracy on the first 1000 test images, loss per step] as float32, and the step-0
gradients of all parameters (in parameters() order) to <out file>.grad0.

usage: python cnn_train.py <resnet18|mobilenetv2> <data dir> <out file> <steps> [mnist|cifar10]
       (steps <= 0: one epoch; evaluation on the first 1000 test images for MNIST, all 10000 for CIFAR-10)
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


def main():
    kind, d, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    data = sys.argv[5] if len(sys.argv) > 5 else "mnist"
    torch.set_num_threads(8)
    xtr, ytr, xte, yte, shape = load(data, d)
    channels = shape[0]
    if steps <= 0:
        steps = len(ytr) // 64
    ntest = 1000 if data == "mnist" else len(yte)
    dev = torch.device(os.environ.get("JITNN_TORCH_DEVICE", "cpu"))   # e.g. cuda: the same model, faster
    net = (resnet18(channels, 10, True, 1) if kind == "resnet18" else mobilenet_v2(channels, 10, True, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = 64
    losses = []
    net.train()
    for b in range(steps):
        x = torch.from_numpy(xtr[b * batch:(b + 1) * batch]).to(dev)
        y = torch.from_numpy(ytr[b * batch:(b + 1) * batch]).to(dev)
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        if b == 0:
            # Step-0 gradients of every parameter, in parameters() order, for the Java gate.
            np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in net.parameters()]).astype(np.float32).tofile(out + ".grad0")
        opt.step()
        losses.append(loss.item())
    net.eval()
    with torch.no_grad():
        pred = np.concatenate([net(torch.from_numpy(xte[i:i + 250]).to(dev)).argmax(1).cpu().numpy() for i in range(0, ntest, 250)])
    acc = float((pred == yte[:ntest]).mean())
    np.array([acc] + losses, dtype=np.float32).tofile(out)
    print(f"{kind}: {steps} steps, loss {losses[0]:.4f} -> {np.mean(losses[-20:]):.4f}, test accuracy ({ntest}) {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
