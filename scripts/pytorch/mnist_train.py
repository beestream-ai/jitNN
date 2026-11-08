# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch reference for jitNN's MNIST gates: the MLP and LeNet-5, from the
same initial weights as jitNN (its Philox init, reproduced in jitnn_init.py), on the same batches
in the same order, with Adam(1e-3) and batch 128. Writes [test accuracy, loss per step] as
float32 for the Java gate tests.

usage: python mnist_train.py <mlp|lenet> <mnist dir> <out file> [epochs]
"""
import sys

import numpy as np
import torch

from jitnn_init import conv2d, linear
from mnist_mlp import load


def init_linear(i, o, seed):
    m = torch.nn.Linear(i, o)
    w, b = linear(i, o, seed)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))
        m.bias.copy_(torch.from_numpy(b))
    return m


def init_conv(i, o, k, seed, padding):
    m = torch.nn.Conv2d(i, o, k, padding=padding)
    w, b = conv2d(i, o, k, seed)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))
        m.bias.copy_(torch.from_numpy(b))
    return m


def model(kind):
    R = torch.nn.ReLU
    if kind == "mlp":
        return torch.nn.Sequential(init_linear(784, 256, 1), R(), init_linear(256, 128, 2), R(), init_linear(128, 10, 3)), (784,)
    if kind == "lenet":
        return torch.nn.Sequential(
            init_conv(1, 6, 5, 1, 2), R(), torch.nn.MaxPool2d(2),
            init_conv(6, 16, 5, 2, 0), R(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            init_linear(400, 120, 3), R(), init_linear(120, 84, 4), R(), init_linear(84, 10, 5)), (1, 28, 28)
    raise ValueError(kind)


def main():
    kind, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else (2 if kind == "mlp" else 1)
    xtr, ytr = load(d, "train")
    xte, yte = load(d, "t10k")
    net, shape = model(kind)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = 128
    losses = []
    for _ in range(epochs):
        for b in range(len(ytr) // batch):
            x = torch.from_numpy(xtr[b * batch:(b + 1) * batch]).reshape(batch, *shape)
            y = torch.from_numpy(ytr[b * batch:(b + 1) * batch])
            loss = torch.nn.functional.cross_entropy(net(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
    with torch.no_grad():
        pred = net(torch.from_numpy(xte).reshape(len(yte), *shape)).argmax(1).numpy()
    acc = float((pred == yte).mean())
    np.array([acc] + losses, dtype=np.float32).tofile(out)
    print(f"{kind}: {len(losses)} steps, loss {losses[0]:.4f} -> {losses[-1]:.4f}, test accuracy {100 * acc:.2f}%")


if __name__ == "__main__":
    main()
