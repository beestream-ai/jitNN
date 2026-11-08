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
"""Writes docs/images/architecture-{light,dark}.svg: jitNN's layers, from user code down to the hardware.

usage: python scripts/docs/architecture.py docs/images
"""
import sys
out_dir = sys.argv[1]
W, H = 1000, 860
THEMES = {
    "light": dict(bg="#FFFFFF", ink="#0B1F33", sub="#4A5B6C", line="#8A99A8", band="#F3F6F9",
                  box="#FFFFFF", stroke="#C9D3DD", accent="#14A884", accentbg="#E6F7F1", tvm="#0B1F33", tvmink="#FFFFFF", tvmsub="#B9C7D4"),
    "dark": dict(bg="#0D1117", ink="#E6EDF3", sub="#9AA7B4", line="#5C6B7A", band="#161B22",
                 box="#0D1117", stroke="#30363D", accent="#22D3A6", accentbg="#0F2A24", tvm="#22D3A6", tvmink="#0B1F33", tvmsub="#0B3B30"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def svg(t):
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">')
    a('<title id="t">jitNN architecture</title>')
    a('<desc id="d">User code and integrations on top of jitnn-nn, the importers and the DJL engine; the core API (Tensor, F, Grad); '
      'the engine that records operators lazily, rewrites and compiles each step into one TornadoVM task graph and replays it; '
      'operators, kernels and vendor libraries; TornadoVM compiling to CUDA, OpenCL and Metal; and the hardware. A CPU backend runs without TornadoVM.</desc>')
    a(f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
      f'<path d="M0,0 L10,5 L0,10 z" fill="{t["line"]}"/></marker></defs>')
    a(f'<rect width="{W}" height="{H}" rx="14" fill="{t["bg"]}"/>')

    def text(x, y, s, size=13, color=None, weight=400, anchor="start", mono=False):
        a(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" font-weight="{weight}" '
          f'fill="{color or t["ink"]}" text-anchor="{anchor}">{esc(s)}</text>')

    def box(x, y, w, h, title, lines=(), code=None, fill=None, stroke=None, tcolor=None, scolor=None):
        a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill or t["box"]}" stroke="{stroke or t["stroke"]}" stroke-width="1.2"/>')
        cy = y + 22
        text(x + 12, cy, title, 14, tcolor, 600)
        if code:
            text(x + w - 12, cy, code, 11, scolor or t["sub"], 400, "end", mono=True)
        for i, l in enumerate(lines):
            text(x + 12, cy + 19 + 17 * i, l, 12, scolor or t["sub"])

    def band(y, h, label):
        a(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="10" fill="{t["band"]}"/>')
        a(f'<text x="34" y="{y + h / 2}" font-family="{FONT}" font-size="11" font-weight="600" fill="{t["sub"]}" '
          f'letter-spacing="1.2" text-anchor="middle" transform="rotate(-90 34 {y + h / 2})">{esc(label)}</text>')

    def arrow(x1, y1, x2, y2):
        a(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["line"]}" stroke-width="1.4" marker-end="url(#arr)"/>')

    L, R = 52, W - 32          # content box inside the bands
    CW = R - L

    # 1. user code
    y = 20
    band(y, 78, "YOUR CODE")
    cols = [("Java applications", ["Tensor and nn APIs"]), ("DJL programs", ["unchanged, jitNN as engine"]),
            ("Pretrained models", ["ONNX, TensorFlow, GGUF, HF"]), ("LLM serving", ["through jitllm"])]
    gw = (CW - 3 * 12) / 4
    for i, (ti, li) in enumerate(cols):
        box(L + i * (gw + 12), y + 10, gw, 58, ti, li)

    # 2. libraries on the core
    y = 112
    band(y, 112, "MODULES")
    mods = [("Neural networks", "jitnn-nn", ["Layers and models, optimisers", "Data loaders, tokenizers", "Classical machine learning"]),
            ("Importers", "jitnn-onnx, -tf", ["ONNX import and export", "TensorFlow via the ONNX", "operators; graph rewrites"]),
            ("Quantised weights", "jitnn-gguf", ["GGUF reader", "Q4_0 to Q8_0 blocks", "jitnn-llm on top"]),
            ("DJL engine", "jitnn-djl", ["NDArray on Tensor", "registered by service", "loader, rank 0"])]
    for i, (ti, code, li) in enumerate(mods):
        box(L + i * (gw + 12), y + 10, gw, 92, ti, li, code)

    # 3. core API
    y = 238
    band(y, 84, "CORE API")
    box(L, y + 10, CW, 64, "jitnn-core", [
        "Tensor, F (operators), Grad (autograd), Autocast (FP16/BF16), Linalg, Fft, SparseMatrix, CustomOp",
        "dist: collectives and data-parallel training over NCCL  ·  io: safetensors and state dicts"], "ai.beestream.jitnn")

    # 4. engine
    y = 336
    band(y, 204, "ENGINE")
    steps = [("1  Record", ["operators lazily,", "per thread"]), ("2  Rewrite", ["fuse epilogues, fold", "BatchNorm, reuse buffers"]),
             ("3  Compile", ["one task graph per step,", "cached by structure"]), ("4  Replay", ["rebind inputs; CUDA", "graphs; weights stay"])]
    sw = (CW - 3 * 26) / 4
    for i, (ti, li) in enumerate(steps):
        x = L + i * (sw + 26)
        box(x, y + 12, sw, 74, ti, li, fill=t["accentbg"], stroke=t["accent"])
        if i < 3:
            arrow(x + sw + 3, y + 49, x + sw + 23, y + 49)
    text(L + CW / 2, y + 108, "at each sync point (item(), toFloatArray(), a loss read): the steps run once, later steps replay", 12, t["sub"], 400, "middle")
    ex = [("TornadoExecutor", ["GPUs through TornadoVM; autotuning", "and per-device tuning tables"]),
          ("CpuExecutor", ["Java Vector API kernels; oneDNN,", "BLAS, BNNS when installed"]),
          ("ReferenceExecutor", ["plain Java, FP64 accumulation:", "the reference the tests check against"])]
    ew = (CW - 2 * 12) / 3
    for i, (ti, li) in enumerate(ex):
        box(L + i * (ew + 12), y + 122, ew, 72, ti, li)

    # 5. operators
    y = 554
    band(y, 112, "OPERATORS")
    ow = (CW - 2 * 12) / 3
    box(L, y + 10, ow, 92, "Operators", ["forward and backward of every", "operator; picks a kernel variant", "or a vendor library"], "ops")
    box(L + ow + 12, y + 10, ow, 92, "Kernels", ["KernelContext kernels: GEMM,", "convolution, attention, norms, ...;", "tensor-core and simdgroup variants"], "kernels")
    box(L + 2 * (ow + 12), y + 10, ow, 92, "Vendor libraries", ["cuBLAS, cuDNN, cuFFT, cuSPARSE,", "CUTLASS, NCCL, MLX as TornadoVM", "library tasks (optional)"], "natives")

    # 6. TornadoVM
    y = 680
    a(f'<rect x="20" y="{y}" width="{W - 40}" height="70" rx="10" fill="{t["tvm"]}"/>')
    text(W / 2, y + 30, "TornadoVM", 17, t["tvmink"], 700, "middle")
    text(W / 2, y + 52, "JIT-compiles the Java kernels for the device: CUDA C (NVRTC), OpenCL C, Metal Shading Language", 12.5, t["tvmsub"] if t is THEMES["dark"] else "#B9C7D4", 400, "middle")

    # 7. hardware
    y = 764
    band(y, 76, "HARDWARE")
    hw = [("NVIDIA GPUs", ["CUDA backend"]), ("AMD, Intel GPUs", ["OpenCL backend"]), ("Apple silicon", ["Metal backend"]), ("CPUs", ["CpuExecutor, or OpenCL"])]
    for i, (ti, li) in enumerate(hw):
        box(L + i * (gw + 12), y + 10, gw, 56, ti, li)
    return "\n".join(o) + "\n</svg>\n"

for name, t in THEMES.items():
    open(f"{out_dir}/architecture-{name}.svg", "w").write(svg(t))
