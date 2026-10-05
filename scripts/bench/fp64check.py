"""Distance of jitNN's and the GPU frameworks' FP32 outputs from an FP64 CPU forward pass.

usage: python fp64check.py <jitnn output dir> <name...>   (names like resnet50_b32, bert_b32_s128)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1  # noqa: E402


def err(a, ref):
    return float(np.max(np.abs(a.astype(np.float64) - ref)) / np.max(np.abs(ref)))


def main(jdir, names):
    torch.set_num_threads(32)
    for name in names:
        kind = "resnet50" if name.startswith("resnet50") else "bert"
        parts = name.split("_")
        b = int(parts[1][1:])
        x = t1.images(b) if kind == "resnet50" else t1.tokens(b, int(parts[2][1:]))
        net = (t1.models.resnet50(3, 1000, False, 1) if kind == "resnet50" else t1.BertEncoder()).eval()
        with torch.no_grad():
            ref = net.double()(torch.from_numpy(x if kind == "bert" else x.astype(np.float64))).numpy()
            gpu = net.float().cuda()(torch.from_numpy(x).cuda()).double().cpu().numpy()
        j = np.fromfile(os.path.join(jdir, name + "_fp32.jitnn.f32"), dtype=np.float32).reshape(ref.shape)
        print(f"[fp64] {name}: jitNN {err(j, ref):.1e}, PyTorch GPU FP32 {err(gpu, ref):.1e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
