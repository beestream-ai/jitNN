"""One framework variant, one shape, one precision on the GPU: median ms host to host (numpy in,
numpy out) and device resident (device tensors in and out), and the max error vs an FP32 reference
(PyTorch eager, TF32 off), on t1.py's models and inputs.

usage: python gsweep.py <t1 dir> <work dir> <framework> <variant> <model> <batch> <seq> <precision>
  framework pytorch: eager | bench | cl | compile | compile-ro | compile-noacg
  framework trt (Torch-TensorRT): dynamo
  framework ort: cuda | exh | exh-graph | opt-exh | opt-exh-graph | trt
  precision: fp32 (TF32 off) | tf32 | fp16
Prints one line starting with [gsweep] and appends a JSON line to <work dir>/gsweep.jsonl."""
import copy
import json
import os
import sys
import time

import numpy as np

t1dir, work, fw, variant, kind, b, t, prec = sys.argv[1:9]
b, t = int(b), int(t)
sys.path.insert(0, t1dir)
import torch  # noqa: E402
import t1  # noqa: E402

WARMUP, RUNS = 10, 50
os.makedirs(work, exist_ok=True)
dev = torch.device("cuda")
name = f"resnet50_b{b}" if kind == "resnet50" else f"bert_b{b}_s{t}"
x_np = t1.images(b) if kind == "resnet50" else t1.tokens(b, t)


def set_tf32(on):
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def build():
    torch.manual_seed(0)
    net = t1.models.resnet50(3, 1000, False, 1) if kind == "resnet50" else t1.BertEncoder()
    return net.eval().float().to(dev)


# FP32 reference, TF32 off, cached per shape
ref_path = os.path.join(work, name + ".ref.npy")
net32 = build()
if not os.path.exists(ref_path):
    set_tf32(False)
    with torch.no_grad():
        np.save(ref_path, net32(torch.from_numpy(x_np).to(dev)).double().cpu().numpy())
ref = np.load(ref_path)
set_tf32(prec == "tf32")
half = prec == "fp16"
net = copy.deepcopy(net32).half() if half else net32
feed = x_np.astype(np.float16) if half and kind == "resnet50" else x_np


def timed(call):
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    ms = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        call()
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return ms[RUNS // 2], ms[RUNS // 10], ms[RUNS * 9 // 10]


host_call = dev_call = None
x_dev = torch.from_numpy(feed).to(dev)
setup_t0 = time.perf_counter()
if fw in ("pytorch", "trt"):
    torch.backends.cudnn.benchmark = variant != "eager"
    if variant == "cl" and kind == "resnet50":
        net = net.to(memory_format=torch.channels_last)
        x_dev = x_dev.contiguous(memory_format=torch.channels_last)
    if fw == "pytorch" and variant.startswith("compile"):
        if kind == "resnet50":
            net = net.to(memory_format=torch.channels_last)
            x_dev = x_dev.contiguous(memory_format=torch.channels_last)
        mode = {"compile": "max-autotune", "compile-ro": "reduce-overhead", "compile-noacg": "max-autotune-no-cudagraphs"}[variant]
        net = torch.compile(net, mode=mode)
    if fw == "trt":
        import torch_tensorrt
        precs = {torch.half} if half else {torch.float32}
        net = torch_tensorrt.compile(net, ir="dynamo", inputs=[x_dev], enabled_precisions=precs, disable_tf32=prec == "fp32",
                                     use_explicit_typing=False, optimization_level=5)
    cl = variant in ("cl",) or variant.startswith("compile")

    def dev_call():
        with torch.inference_mode():
            y = net(x_dev)
        torch.cuda.synchronize()
        return y

    def host_call():
        with torch.inference_mode():
            x = torch.from_numpy(feed).to(dev)
            if cl and kind == "resnet50":
                x = x.contiguous(memory_format=torch.channels_last)
            return net(x).cpu().numpy()
else:
    import onnxruntime as ort
    path = os.path.join(work, f"{name}_{'fp16' if half else 'fp32'}.onnx")
    if not os.path.exists(path):
        t1.export(net, x_dev, path)
    if variant.startswith("opt"):
        from onnxruntime.transformers import optimizer
        opath = path.replace(".onnx", ".opt.onnx")
        if not os.path.exists(opath):
            mt = "bert" if kind == "bert" else "unet"
            o = optimizer.optimize_model(path, model_type=mt, num_heads=12 if kind == "bert" else 0, hidden_size=768 if kind == "bert" else 0,
                                         opt_level=1, use_gpu=True)
            print("[gsweep] fused ops", o.get_fused_operator_statistics(), flush=True)
            o.save_model_to_file(opath, use_external_data_format=True)
        path = opath
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    graph = variant.endswith("graph")
    if variant == "trt":
        cache = os.path.join(work, "trt-cache")
        os.makedirs(cache, exist_ok=True)
        provs = [("TensorrtExecutionProvider", {"trt_fp16_enable": half, "trt_engine_cache_enable": True, "trt_engine_cache_path": cache,
                                                "trt_builder_optimization_level": 5, "trt_timing_cache_enable": True}),
                 ("CUDAExecutionProvider", {"use_tf32": int(prec != "fp32")})]
    else:
        cuda = {"use_tf32": int(prec != "fp32"), "cudnn_conv_algo_search": "DEFAULT" if variant == "cuda" else "EXHAUSTIVE",
                "enable_cuda_graph": int(graph), "cudnn_conv_use_max_workspace": 1, "do_copy_in_default_stream": 1}
        provs = [("CUDAExecutionProvider", cuda)]
    sess = ort.InferenceSession(path, opts, providers=provs)
    used = sess.get_providers()
    assert used[0] == provs[0][0], used
    out_meta = sess.get_outputs()[0]
    x_ort = ort.OrtValue.ortvalue_from_numpy(feed, "cuda", 0)
    io = sess.io_binding()
    io.bind_ortvalue_input("x", x_ort)
    io.bind_output(out_meta.name, "cuda", 0)
    if graph:
        # CUDA graphs need fixed device buffers for the output too
        sess.run_with_iobinding(io)
        y_ort = io.get_outputs()[0]
        io = sess.io_binding()
        io.bind_ortvalue_input("x", x_ort)
        io.bind_ortvalue_output(out_meta.name, y_ort)

    def dev_call():
        sess.run_with_iobinding(io)
        io.synchronize_outputs()
        return io

    def host_call():
        x_ort.update_inplace(feed)
        sess.run_with_iobinding(io)
        return io.get_outputs()[0].numpy()

out = host_call()
setup_s = time.perf_counter() - setup_t0
err = float(np.max(np.abs(out.astype(np.float64) - ref)) / max(np.max(np.abs(ref)), 1e-30))
hmed, hp10, hp90 = timed(host_call)
dmed, dp10, dp90 = timed(dev_call)
mem = torch.cuda.max_memory_allocated() / 2**20
row = dict(fw=fw, variant=variant, model=name, prec=prec, host_ms=hmed, host_p10=hp10, host_p90=hp90, dev_ms=dmed, dev_p10=dp10, dev_p90=dp90,
           err=err, setup_s=setup_s, torch_mem_mb=mem, torch=torch.__version__)
with open(os.path.join(work, "gsweep.jsonl"), "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"[gsweep] {fw:7s} {variant:14s} {name:16s} {prec:4s} host {hmed:9.3f} ms (p10 {hp10:8.3f} p90 {hp90:8.3f})  dev {dmed:9.3f} ms  err {err:.1e}  setup {setup_s:6.1f} s", flush=True)
