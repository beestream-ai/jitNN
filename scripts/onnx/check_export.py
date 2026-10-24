"""Runs the models OnnxExportTest wrote (-Djitnn.onnx.exportDir=<dir>) in ONNX Runtime and
compares with jitNN's outputs: <name>.onnx, <name>.in<i>.bin (the inputs, raw, in the graph
inputs' types) and <name>.out.f32.

usage: python check_export.py <dir>
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

d = sys.argv[1]
worst = 0.0
for f in sorted(os.listdir(d)):
    if not f.endswith(".onnx"):
        continue
    name = f[:-5]
    path = os.path.join(d, f)
    onnx.checker.check_model(path, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {}
    for i, inp in enumerate(sess.get_inputs()):
        dtype = np.int32 if inp.type == "tensor(int32)" else np.float32
        feeds[inp.name] = np.fromfile(os.path.join(d, f"{name}.in{i}.bin"), dtype=dtype).reshape(inp.shape)
    got = sess.run(None, feeds)[0].ravel()
    want = np.fromfile(os.path.join(d, name + ".out.f32"), dtype=np.float32)
    rel = float(np.abs(got - want).max() / max(np.abs(want).max(), 1e-6))
    worst = max(worst, rel)
    print(f"[export] {name:22s} ONNX Runtime vs jitNN: max error {rel:.1e} of the largest value", flush=True)
sys.exit(0 if worst <= 1e-4 else 1)
