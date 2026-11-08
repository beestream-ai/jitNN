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
"""PyTorch reference for jitNN's MNIST gate (MnistAccelTest): same weights (jitNN Philox init),
same batch order, same normalisation and the same Adam settings. Writes the per-step losses and
the test accuracy so the Java test can compare against them.

usage: python mnist_mlp.py <mnist dir> <out file> [epochs]
"""
import gzip
import struct
import sys

import numpy as np
import torch

from jitnn_init import linear


def read(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic == 2051:
            rows, cols = struct.unpack(">II", f.read(8))
            raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
            return ((raw.astype(np.float32) / np.float32(255.0) - np.float32(0.1307)) / np.float32(0.3081)).astype(np.float32)
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load(d, prefix):
    import os
    def pick(name):
        p = os.path.join(d, name)
        return p if os.path.exists(p) else p + ".gz"
    return read(pick(prefix + "-images-idx3-ubyte")), read(pick(prefix + "-labels-idx1-ubyte"))


def main():
    d, out = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    torch.manual_seed(0)
    xtr, ytr = load(d, "train")
    xte, yte = load(d, "t10k")

    layers = []
    for (i, o, seed) in [(784, 256, 1), (256, 128, 2), (128, 10, 3)]:
        lin = torch.nn.Linear(i, o)
        w, b = linear(i, o, seed)
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(w))
            lin.bias.copy_(torch.from_numpy(b))
        layers += [lin, torch.nn.ReLU()]
    model = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    batch = 128
    losses = []
    for _ in range(epochs):
        for b in range(len(ytr) // batch):
            x = torch.from_numpy(xtr[b * batch:(b + 1) * batch])
            y = torch.from_numpy(ytr[b * batch:(b + 1) * batch])
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

    with torch.no_grad():
        pred = model(torch.from_numpy(xte)).argmax(1).numpy()
    accuracy = float((pred == yte).mean())
    np.array([accuracy] + losses, dtype=np.float32).tofile(out)
    print(f"pytorch {torch.__version__}: {len(losses)} steps, first loss {losses[0]:.6f}, last {losses[-1]:.6f}, test accuracy {100 * accuracy:.2f}%")


if __name__ == "__main__":
    main()
