"""PyTorch reference for jitNN's CNN gates: ResNet-18 or MobileNetV2 (small variants) on MNIST,
from jitNN's initial weights, Adam(1e-3), batch 64, the first <steps> training batches in order.
Writes [test accuracy on the first 1000 test images, loss per step] as float32, and the step-0
gradients of all parameters (in parameters() order) to <out file>.grad0.

usage: python cnn_train.py <resnet18|mobilenetv2> <mnist dir> <out file> <steps>
"""
import sys

import numpy as np
import torch

from mnist_mlp import load
from models import mobilenet_v2, resnet18


def main():
    kind, d, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    torch.set_num_threads(8)
    xtr, ytr = load(d, "train")
    xte, yte = load(d, "t10k")
    net = resnet18(1, 10, True, 1) if kind == "resnet18" else mobilenet_v2(1, 10, True, 1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = 64
    losses = []
    net.train()
    for b in range(steps):
        x = torch.from_numpy(xtr[b * batch:(b + 1) * batch]).reshape(batch, 1, 28, 28)
        y = torch.from_numpy(ytr[b * batch:(b + 1) * batch])
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        if b == 0:
            # Step-0 gradients of every parameter, in parameters() order, for the Java gate.
            np.concatenate([p.grad.detach().numpy().ravel() for p in net.parameters()]).astype(np.float32).tofile(out + ".grad0")
        opt.step()
        losses.append(loss.item())
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(xte[:1000]).reshape(1000, 1, 28, 28)).argmax(1).numpy()
    acc = float((pred == yte[:1000]).mean())
    np.array([acc] + losses, dtype=np.float32).tofile(out)
    print(f"{kind}: {steps} steps, loss {losses[0]:.4f} -> {np.mean(losses[-20:]):.4f}, test accuracy (1000) {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
