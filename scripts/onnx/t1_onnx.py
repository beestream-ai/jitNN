"""ONNX Runtime on the T1 ONNX files (written by jitNN-benchmarking's scripts/t1.py): the reference outputs and
timings that OnnxT1AccelTest compares jitNN against, on the same files and inputs.

For each <work dir>/<name>.onnx: inputs as InferenceBenchAccelTest builds them, ONNX Runtime's CUDA
provider (TF32 off for FP32 files), host input in and host output out, 10 warm-up calls and the
median of 50. Writes <name>.ort.f32 (the output) and ort_t1.json.

usage: python t1_onnx.py <work dir>
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

VOCAB = 30522
work = sys.argv[1]


def images(b):
    i = np.arange(b * 3 * 224 * 224, dtype=np.int64)
    return (((i * 2654435761) % 1000).astype(np.float32) / np.float32(500.0) - np.float32(1.0)).reshape(b, 3, 224, 224)


def tokens(b, t):
    i = np.arange(b * t, dtype=np.int64)
    return ((i * 7919 + 13) % VOCAB).reshape(b, t)


results = []
for f in sorted(os.listdir(work)):
    if not f.endswith(".onnx"):
        continue
    name = f[:-5]
    parts = name.split("_")
    fp16 = name.endswith("fp16")
    if name.startswith("resnet50"):
        b = int(parts[1][1:])
        feed = images(b).astype(np.float16 if fp16 else np.float32)
    else:
        b, t = int(parts[1][1:]), int(parts[2][1:])
        feed = tokens(b, t)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(os.path.join(work, f), opts, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "EXHAUSTIVE", "use_tf32": 0 if not fp16 else 1})])
    call = lambda: sess.run(None, {sess.get_inputs()[0].name: feed})[0]  # noqa: E731
    out = call()
    for _ in range(9):
        call()
    ms = []
    for _ in range(50):
        t0 = time.perf_counter()
        call()
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    out.astype(np.float32).tofile(os.path.join(work, name + ".ort.f32"))
    results.append(dict(run=name, median_ms=ms[25], p10_ms=ms[5], p90_ms=ms[45]))
    print(f"[t1-onnx] onnxruntime {name:24s} median {ms[25]:8.3f} ms", flush=True)
    del sess
with open(os.path.join(work, "ort_t1.json"), "w") as fh:
    json.dump(dict(onnxruntime=ort.__version__, results=results), fh, indent=1)
