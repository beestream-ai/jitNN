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
"""MNIST and CIFAR-10 loaders matching jitnn.data (same normalisation, same order)."""
import os

import numpy as np

from mnist_mlp import load as load_mnist

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)


def load_cifar10(d, train):
    base = os.path.join(d, "cifar-10-batches-bin") if os.path.isdir(os.path.join(d, "cifar-10-batches-bin")) else d
    files = [f"data_batch_{i}.bin" for i in range(1, 6)] if train else ["test_batch.bin"]
    raw = np.concatenate([np.fromfile(os.path.join(base, f), dtype=np.uint8).reshape(10000, 3073) for f in files])
    y = raw[:, 0].astype(np.int64)
    x = raw[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32) / np.float32(255.0)
    x = ((x - CIFAR_MEAN) / CIFAR_STD).astype(np.float32)
    return x, y


def load(name, d):
    """(xtrain, ytrain, xtest, ytest, sample shape) with x as float32 [N, C, H, W]."""
    if name == "mnist":
        xtr, ytr = load_mnist(d, "train")
        xte, yte = load_mnist(d, "t10k")
        return xtr.reshape(-1, 1, 28, 28), ytr, xte.reshape(-1, 1, 28, 28), yte, (1, 28, 28)
    if name == "cifar10":
        xtr, ytr = load_cifar10(d, True)
        xte, yte = load_cifar10(d, False)
        return xtr, ytr, xte, yte, (3, 32, 32)
    raise ValueError(name)
