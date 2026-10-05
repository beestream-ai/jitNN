"""TorchScript versions of t1.py's models, for DJL's PyTorch engine: one traced file per configuration.

usage: python trace.py <work dir>
"""
import copy
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1  # noqa: E402


def main(work):
    os.makedirs(work, exist_ok=True)
    dev = torch.device("cuda")
    configs = [("resnet50", b, None) for b in t1.RESNET_BATCHES] + [("bert", b, s) for b, s in t1.BERT_SHAPES]
    nets = {}
    for kind, b, s in configs:
        if kind not in nets:
            nets[kind] = (t1.models.resnet50(3, 1000, False, 1) if kind == "resnet50" else t1.BertEncoder()).eval()
        base = f"resnet50_b{b}" if kind == "resnet50" else f"bert_b{b}_s{s}"
        for dtype in ("fp32", "fp16"):
            net = nets[kind].to(dev) if dtype == "fp32" else copy.deepcopy(nets[kind]).half().to(dev)
            x = torch.from_numpy(t1.images(b) if kind == "resnet50" else t1.tokens(b, s)).to(dev)
            if kind == "resnet50" and dtype == "fp16":
                x = x.half()
            path = os.path.join(work, f"{base}_{dtype}.pt")
            if not os.path.exists(path):
                with torch.no_grad():
                    torch.jit.trace(net, (x,)).save(path)
            print("traced", path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
