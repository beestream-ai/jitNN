"""INT8 versions of the T1 ONNX files, made with ONNX Runtime's quantisation tools, and their
references for OnnxT1AccelTest (-Djitnn.onnx.t1.match=int8).

- BERT: dynamic quantisation (int8 weights, activations quantised per call: DynamicQuantizeLinear
  and MatMulInteger), as quantize_dynamic writes it.
- ResNet-50: static quantisation in the QDQ format, int8 per-channel weights, uint8 activations
  calibrated on the T1 images.

For each model: <name>.ort.f32, the output of ONNX Runtime's CPU provider (which runs the integer
kernels), the agreement of that output with the FP32 model, and the median time of ONNX Runtime's
CUDA provider on the INT8 file. Inputs as in t1_onnx.py.

usage: python int8_onnx.py <T1 work dir> <out dir>
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static

VOCAB = 30522
work, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)


def images(b, seed=0):
    i = np.arange(b * 3 * 224 * 224, dtype=np.int64) + seed * 7777
    return (((i * 2654435761) % 1000).astype(np.float32) / np.float32(500.0) - np.float32(1.0)).reshape(b, 3, 224, 224)


def tokens(b, t):
    i = np.arange(b * t, dtype=np.int64)
    return ((i * 7919 + 13) % VOCAB).reshape(b, t)


class Images(CalibrationDataReader):
    def __init__(self, name, b, batches=8):
        self.feeds = iter([{name: images(b, seed=s)} for s in range(1, batches + 1)])

    def get_next(self):
        return next(self.feeds, None)


def run(path, feed, providers):
    sess = ort.InferenceSession(path, providers=providers)
    name = sess.get_inputs()[0].name
    call = lambda: sess.run(None, {name: feed})[0]  # noqa: E731
    first = call()
    for _ in range(9):
        call()
    ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        call()
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return first, ms[25]


results = []
for src, dst, b in (("bert_b1_s128_fp32", "bert_b1_s128_int8dyn", 1), ("bert_b32_s128_fp32", "bert_b32_s128_int8dyn", 32),
                    ("resnet50_b1_fp32", "resnet50_b1_int8qdq", 1), ("resnet50_b32_fp32", "resnet50_b32_int8qdq", 32)):
    fp32 = os.path.join(work, src + ".onnx")
    path = os.path.join(out, dst + ".onnx")
    if dst.startswith("bert"):
        feed = tokens(b, 128)
        if not os.path.exists(path):
            quantize_dynamic(fp32, path, weight_type=QuantType.QInt8)
    else:
        feed = images(b)
        if not os.path.exists(path):
            input_name = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).get_inputs()[0].name
            quantize_static(fp32, path, Images(input_name, b), quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    want, _ = run(fp32, feed, ["CPUExecutionProvider"])
    got, cpu_ms = run(path, feed, ["CPUExecutionProvider"])
    got.astype(np.float32).tofile(os.path.join(out, dst + ".ort.f32"))
    _, cuda_ms = run(path, feed, [("CUDAExecutionProvider", {"use_tf32": 0}), "CPUExecutionProvider"])
    rel = float(np.abs(got - want).max() / np.abs(want).max())
    top1 = float((got.reshape(b, -1).argmax(1) == want.reshape(b, -1).argmax(1)).mean()) if dst.startswith("resnet") else None
    results.append(dict(run=dst, ort_cpu_ms=cpu_ms, ort_cuda_ms=cuda_ms, rel_err_vs_fp32=rel, top1_agreement=top1))
    print(f"[int8] {dst:24s} ORT CPU {cpu_ms:8.2f} ms  ORT CUDA {cuda_ms:8.2f} ms  vs FP32 max rel err {rel:.2e}  top-1 {top1}", flush=True)
with open(os.path.join(out, "ort_int8.json"), "w") as fh:
    json.dump(dict(onnxruntime=ort.__version__, results=results), fh, indent=1)
