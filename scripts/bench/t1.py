"""Inference comparison of jitNN with PyTorch and ONNX Runtime on the same GPU: ResNet-50 and BERT-base,
built from jitNN's seeds (bit-identical weights), on the inputs InferenceBenchAccelTest uses.

Protocol for every framework: host input in, host output out, 10 warm-up calls, then the median
of 50 timed calls. FP32 runs have TF32 off everywhere (PyTorch and ONNX Runtime's CUDA provider),
so they compare FP32 arithmetic; FP16 runs use FP16 weights and activations.

usage: python t1.py [--device cuda|mps|cpu] [--threads N] [--models resnet50,bert] <work dir> [jitnn output dir]

--device mps runs PyTorch on Apple's Metal (MPS) and ONNX Runtime on its CoreML provider, limited
to the CPU and GPU (not the Neural Engine), so both use the same GPU as jitNN's Metal backend.

--device cpu runs PyTorch eager and ONNX Runtime's CPU provider, FP32 only, on --threads threads
(default: all the cores PyTorch sees); the comparison for jitNN's CPU backend.

Exports one ONNX model per configuration into <work dir>, prints one line per run and writes
<work dir>/t1.json. With a jitNN output dir (written by InferenceBenchAccelTest with
-Djitnn.bench.out) it also reports how far jitNN's outputs are from PyTorch's FP32 ones.
"""
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pytorch"))
import models  # noqa: E402
import transformers  # noqa: E402


torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

WARMUP, RUNS = int(os.environ.get("T1_WARMUP", 10)), int(os.environ.get("T1_RUNS", 50))
VOCAB = 30522
RESNET_BATCHES = [1, 32, 128]
BERT_SHAPES = [(1, 128), (32, 128), (8, 512)]


class BertEncoder(torch.nn.Module):
    """transformers.Bert's encoder: hidden states [B, T, 768], no padding mask (jitNN's Bert.forward)."""

    def __init__(self):
        super().__init__()
        self.bert = transformers.Bert(VOCAB, 512, 768, 12, 12, 1)

    def forward(self, idx):
        b = self.bert
        x = b.emb_norm(b.tokens[idx] + b.positions[: idx.shape[1]])
        for block in b.blocks:
            x = block(x, None)
        return x


def images(b):
    i = np.arange(b * 3 * 224 * 224, dtype=np.int64)
    return (((i * 2654435761) % 1000).astype(np.float32) / np.float32(500.0) - np.float32(1.0)).reshape(b, 3, 224, 224)


def tokens(b, t):
    i = np.arange(b * t, dtype=np.int64)
    return ((i * 7919 + 13) % VOCAB).reshape(b, t)


def timed(call):
    first = call()
    for _ in range(WARMUP - 1):
        call()
    ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        call()
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return first, ms[RUNS // 2], ms[RUNS // 10], ms[RUNS * 9 // 10]


def export(model, x, path):
    if not os.path.exists(path):
        torch.onnx.export(model, (x,), path, input_names=["x"], output_names=["y"], opset_version=18, dynamo=True, external_data=True)
    return path


def ort_session(path, fp32, device):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if device == "cpu":
        opts.intra_op_num_threads = torch.get_num_threads()
        return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    if device == "mps":
        coreml = {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU", "RequireStaticInputShapes": "1"}
        return ort.InferenceSession(path, opts, providers=[("CoreMLExecutionProvider", coreml), "CPUExecutionProvider"])
    cuda = {"device_id": 0, "cudnn_conv_algo_search": "EXHAUSTIVE", "use_tf32": 0 if fp32 else 1}
    return ort.InferenceSession(path, opts, providers=[("CUDAExecutionProvider", cuda)])


ORT_PROVIDER = {"cuda": "CUDAExecutionProvider", "mps": "CoreMLExecutionProvider", "cpu": "CPUExecutionProvider"}


def gpu_name(device):
    if device == "cpu":
        import platform
        import subprocess
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")), platform.machine())
        return f"{model}, {torch.get_num_threads()} threads"
    if device == "mps":
        import platform
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        return chip or platform.machine()
    return torch.cuda.get_device_name(0)


def main(work, jitnn_dir, device="cuda", kinds=("resnet50", "bert")):
    os.makedirs(work, exist_ok=True)
    dev = torch.device(device)
    results = []

    def record(engine, name, samples, stats, out, ref):
        first, med, p10, p90 = stats
        err = None
        if ref is not None:
            err = float(np.max(np.abs(out.astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-30))
        results.append(dict(engine=engine, run=name, median_ms=med, p10_ms=p10, p90_ms=p90, samples_per_s=samples * 1000 / med, max_rel_err_vs_fp32=err))
        e = "" if err is None else f"  max err vs PyTorch FP32 {err:.1e}"
        print(f"[t1] {engine:12s} {name:28s} median {med:8.3f} ms  p10 {p10:8.3f}  p90 {p90:8.3f}  {samples * 1000 / med:10.1f} samples/s{e}", flush=True)

    def jitnn_check(name, ref):
        if jitnn_dir is None:
            return
        f = os.path.join(jitnn_dir, name + ".jitnn.f32")
        if os.path.exists(f):
            out = np.fromfile(f, dtype=np.float32).reshape(ref.shape)
            err = float(np.max(np.abs(out.astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-30))
            print(f"[t1] jitnn output {name}: max err vs PyTorch FP32 {err:.1e}", flush=True)
            results.append(dict(engine="jitnn-check", run=name, max_rel_err_vs_fp32=err))

    configs = [("resnet50", b, None) for b in RESNET_BATCHES] + [("bert", b, t) for b, t in BERT_SHAPES]
    configs = [c for c in configs if c[0] in kinds]
    nets = {}
    for kind, b, t in configs:
        if kind not in nets:
            net = models.resnet50(3, 1000, False, 1) if kind == "resnet50" else BertEncoder()
            nets[kind] = net.eval()
        net32 = nets[kind].float().to(dev)
        x_np = images(b) if kind == "resnet50" else tokens(b, t)
        base = f"resnet50_b{b}" if kind == "resnet50" else f"bert_b{b}_s{t}"
        with torch.no_grad():
            ref = net32(torch.from_numpy(x_np).to(dev)).cpu().double().numpy()
        jitnn_check(base + "_fp32", ref)
        if device != "cpu":
            jitnn_check(base + "_fp16", ref)
        for dtype in ("fp32",) if device == "cpu" else ("fp32", "fp16"):
            fp32 = dtype == "fp32"
            # a copy: converting the shared model to FP16 and back would round its weights
            net = net32 if fp32 else copy.deepcopy(net32).half()
            feed = x_np if kind == "bert" or fp32 else x_np.astype(np.float16)
            name = f"{base}_{dtype}"

            def torch_call():
                with torch.no_grad():
                    return net(torch.from_numpy(feed).to(dev)).cpu().numpy()

            record("pytorch", name, b, timed(torch_call), torch_call(), ref)
            path = export(net, torch.from_numpy(feed).to(dev), os.path.join(work, name + ".onnx"))
            sess = ort_session(path, fp32, device)
            assert ORT_PROVIDER[device] in sess.get_providers(), sess.get_providers()
            ort_call = lambda: sess.run(None, {"x": feed})[0]  # noqa: E731
            record("onnxruntime", name, b, timed(ort_call), ort_call(), ref)
            del sess
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    import onnxruntime as ort
    with open(os.path.join(work, "t1.json"), "w") as f:
        json.dump(dict(torch=torch.__version__, onnxruntime=ort.__version__, gpu=gpu_name(device), results=results), f, indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ResNet-50 and BERT-base inference with PyTorch and ONNX Runtime")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for PyTorch and ONNX Runtime (default: PyTorch's)")
    ap.add_argument("--models", default="resnet50,bert")
    ap.add_argument("work")
    ap.add_argument("jitnn_dir", nargs="?")
    a = ap.parse_args()
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    main(a.work, a.jitnn_dir, a.device, tuple(a.models.split(",")))
