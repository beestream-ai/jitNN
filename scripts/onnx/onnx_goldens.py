"""ONNX goldens for jitnn-onnx: small models, one per operator (or pattern), with outputs from
ONNX Runtime's CPU provider, plus two models exported from PyTorch (a CNN and a transformer block).

Writes into <out dir>: <case>.onnx for every case, and
  manifest.txt   one line per array: <case> <in|out>:<name> <f32|i32> <offset> <count> <dims...>
  data.bin       the arrays, little-endian, back to back

usage: python onnx_goldens.py <out dir>
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

rng = np.random.default_rng(7)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
cases = []  # (name, model, feeds)


def rnd(*s, lo=-2.0, hi=2.0):
    return rng.uniform(lo, hi, size=s).astype(np.float32)


def model(name, nodes, inputs, outputs, inits=(), opset=18):
    """inputs: [(name, array)], outputs: [names]; initializers: [(name, array)]."""
    vi = []
    for n, a in inputs:
        et = TensorProto.INT64 if a.dtype == np.int64 else TensorProto.FLOAT
        vi.append(helper.make_tensor_value_info(n, et, list(a.shape)))
    g = helper.make_graph(nodes, name, vi, [], [numpy_helper.from_array(a, n) for n, a in inits])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", opset)])
    m.ir_version = 9
    # graph outputs need their element types: take them from shape inference
    inferred = {v.name: v for v in onnx.shape_inference.infer_shapes(m).graph.value_info}
    for o in outputs:
        m.graph.output.append(inferred[o])
    cases.append((name, m, dict(inputs)))


N = helper.make_node
# ---- element-wise with broadcasting
for op in ["Add", "Sub", "Mul", "Div", "Max", "Min"]:
    b = rnd(1, 5, lo=0.5, hi=2) if op == "Div" else rnd(1, 5)
    model(op.lower(), [N(op, ["a", "b"], ["y"])], [("a", rnd(3, 4, 5)), ("b", b)], ["y"])
model("pow_scalar", [N("Pow", ["a", "e"], ["y"])], [("a", rnd(4, 6, lo=0.1, hi=2))], ["y"], [("e", np.array(3.0, np.float32))])
model("pow_square", [N("Pow", ["a", "e"], ["y"])], [("a", rnd(4, 6))], ["y"], [("e", np.array(2.0, np.float32))])
for op, lo in [("Relu", -2), ("Sigmoid", -2), ("Tanh", -2), ("Erf", -2), ("Exp", -2), ("Log", 0.1), ("Sqrt", 0.1), ("Abs", -2), ("Neg", -2), ("Softplus", -2),
               ("Reciprocal", 0.2), ("Floor", -2), ("Ceil", -2), ("Sin", -2), ("Cos", -2)]:
    model(op.lower(), [N(op, ["x"], ["y"])], [("x", rnd(5, 7, lo=lo))], ["y"])
model("leaky_relu", [N("LeakyRelu", ["x"], ["y"], alpha=0.2)], [("x", rnd(5, 7))], ["y"])
model("elu", [N("Elu", ["x"], ["y"], alpha=0.7)], [("x", rnd(5, 7))], ["y"])
model("hard_sigmoid", [N("HardSigmoid", ["x"], ["y"], alpha=0.25, beta=0.4)], [("x", rnd(5, 7, lo=-4, hi=4))], ["y"])
model("gelu", [N("Gelu", ["x"], ["y"])], [("x", rnd(5, 7))], ["y"], opset=20)
model("gelu_tanh", [N("Gelu", ["x"], ["y"], approximate="tanh")], [("x", rnd(5, 7))], ["y"], opset=20)
model("clip", [N("Clip", ["x", "lo", "hi"], ["y"])], [("x", rnd(5, 7))], ["y"], [("lo", np.array(-0.5, np.float32)), ("hi", np.array(1.2, np.float32))])
model("prelu", [N("PRelu", ["x", "s"], ["y"])], [("x", rnd(2, 3, 4, 4))], ["y"], [("s", rnd(3, 1, 1, lo=0, hi=0.5))])
model("where_less", [N("Less", ["a", "b"], ["c"]), N("Where", ["c", "a", "b"], ["y"])], [("a", rnd(4, 5)), ("b", rnd(4, 5))], ["y"])
model("equal_cast", [N("Equal", ["a", "b"], ["c"]), N("Cast", ["c"], ["y"], to=TensorProto.FLOAT)], [("a", rnd(4, 5).round()), ("b", rnd(4, 5).round())], ["y"])
# ---- matrix products
model("matmul_3d_2d", [N("MatMul", ["a", "w"], ["y"])], [("a", rnd(2, 7, 9))], ["y"], [("w", rnd(9, 5))])
model("matmul_4d", [N("MatMul", ["a", "b"], ["y"])], [("a", rnd(2, 3, 6, 8)), ("b", rnd(2, 3, 8, 4))], ["y"])
model("matmul_bcast", [N("MatMul", ["a", "b"], ["y"])], [("a", rnd(1, 6, 8)), ("b", rnd(3, 8, 4))], ["y"])
model("gemm_transb", [N("Gemm", ["a", "w", "c"], ["y"], transB=1)], [("a", rnd(6, 10))], ["y"], [("w", rnd(4, 10)), ("c", rnd(4))])
model("gemm_alpha_beta", [N("Gemm", ["a", "w", "c"], ["y"], alpha=0.5, beta=2.0, transA=1)], [("a", rnd(10, 6))], ["y"], [("w", rnd(10, 4)), ("c", rnd(6, 4))])
# ---- convolution and pooling
model("conv", [N("Conv", ["x", "w", "b"], ["y"], pads=[1, 1, 1, 1], strides=[1, 1])], [("x", rnd(2, 3, 9, 9))], ["y"], [("w", rnd(4, 3, 3, 3)), ("b", rnd(4))])
model("conv_stride_groups", [N("Conv", ["x", "w"], ["y"], pads=[1, 1, 1, 1], strides=[2, 2], group=2)], [("x", rnd(1, 4, 10, 10))], ["y"], [("w", rnd(6, 2, 3, 3))])
model("conv_asym_pads", [N("Conv", ["x", "w"], ["y"], pads=[0, 1, 1, 0])], [("x", rnd(1, 2, 7, 7))], ["y"], [("w", rnd(3, 2, 3, 3))])
model("conv_same", [N("Conv", ["x", "w"], ["y"], auto_pad="SAME_UPPER")], [("x", rnd(1, 2, 8, 8))], ["y"], [("w", rnd(3, 2, 3, 3))])
model("conv1d", [N("Conv", ["x", "w", "b"], ["y"], pads=[2, 2], dilations=[2])], [("x", rnd(2, 3, 20))], ["y"], [("w", rnd(5, 3, 5)), ("b", rnd(5))])
model("conv_transpose", [N("ConvTranspose", ["x", "w", "b"], ["y"], strides=[2, 2], pads=[1, 1, 1, 1], output_padding=[1, 1])], [("x", rnd(1, 4, 5, 5))], ["y"],
      [("w", rnd(4, 3, 3, 3)), ("b", rnd(3))])
model("max_pool", [N("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])], [("x", rnd(2, 3, 11, 11))], ["y"])
model("avg_pool_exclude_pad", [N("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1])], [("x", rnd(1, 2, 6, 6))], ["y"])
model("avg_pool_include_pad", [N("AveragePool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2], pads=[1, 1, 1, 1], count_include_pad=1)], [("x", rnd(1, 2, 6, 6))], ["y"])
model("global_avg_pool", [N("GlobalAveragePool", ["x"], ["y"])], [("x", rnd(2, 5, 7, 7))], ["y"])
# ---- normalisation and softmax
model("batch_norm", [N("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], epsilon=1e-3)], [("x", rnd(2, 3, 5, 5))], ["y"],
      [("s", rnd(3)), ("b", rnd(3)), ("m", rnd(3)), ("v", rnd(3, lo=0.5, hi=2))])
model("layer_norm", [N("LayerNormalization", ["x", "s", "b"], ["y"], epsilon=1e-5)], [("x", rnd(2, 7, 16))], ["y"], [("s", rnd(16)), ("b", rnd(16))])
model("layer_norm_axis", [N("LayerNormalization", ["x", "s", "b"], ["y"], axis=-2)], [("x", rnd(3, 4, 5))], ["y"], [("s", rnd(4, 5)), ("b", rnd(4, 5))])
model("instance_norm", [N("InstanceNormalization", ["x", "s", "b"], ["y"])], [("x", rnd(2, 3, 6, 6))], ["y"], [("s", rnd(3)), ("b", rnd(3))])
model("softmax_axis1", [N("Softmax", ["x"], ["y"], axis=1)], [("x", rnd(3, 6, 4))], ["y"])
model("log_softmax", [N("LogSoftmax", ["x"], ["y"])], [("x", rnd(3, 9))], ["y"])
# ---- reductions
model("reduce_mean", [N("ReduceMean", ["x", "ax"], ["y"], keepdims=0)], [("x", rnd(3, 4, 5))], ["y"], [("ax", np.array([1], np.int64))])
model("reduce_sum_keep", [N("ReduceSum", ["x", "ax"], ["y"])], [("x", rnd(3, 4, 5))], ["y"], [("ax", np.array([0, 2], np.int64))])
model("reduce_max_all", [N("ReduceMax", ["x"], ["y"], keepdims=0)], [("x", rnd(3, 4, 5))], ["y"])
model("cumsum", [N("CumSum", ["x", "ax"], ["y"])], [("x", rnd(4, 6))], ["y"], [("ax", np.array(1, np.int64))])
# ---- shapes
model("reshape_zero_minus1", [N("Reshape", ["x", "s"], ["y"])], [("x", rnd(2, 3, 4))], ["y"], [("s", np.array([0, -1], np.int64))])
model("transpose", [N("Transpose", ["x"], ["y"], perm=[2, 0, 1])], [("x", rnd(2, 3, 4))], ["y"])
model("squeeze_unsqueeze", [N("Unsqueeze", ["x", "a"], ["u"]), N("Squeeze", ["u", "a"], ["y0"]), N("Relu", ["u"], ["y1"])], [("x", rnd(3, 4))], ["y0", "y1"],
      [("a", np.array([0, 3], np.int64))])
model("flatten", [N("Flatten", ["x"], ["y"], axis=2)], [("x", rnd(2, 3, 4, 5))], ["y"])
model("concat", [N("Concat", ["a", "b"], ["y"], axis=1)], [("a", rnd(2, 3, 4)), ("b", rnd(2, 2, 4))], ["y"])
model("split", [N("Split", ["x", "s"], ["y0", "y1", "y2"], axis=1)], [("x", rnd(2, 9))], ["y0", "y1", "y2"], [("s", np.array([2, 3, 4], np.int64))])
model("slice_neg", [N("Slice", ["x", "st", "en", "ax"], ["y"])], [("x", rnd(5, 8))], ["y"],
      [("st", np.array([1, -5], np.int64)), ("en", np.array([4, -1], np.int64)), ("ax", np.array([0, 1], np.int64))])
model("gather_indices", [N("Gather", ["x", "i"], ["y"], axis=1)], [("x", rnd(3, 6, 2))], ["y"], [("i", np.array([[4, 0], [1, 1]], np.int64))])
model("gather_scalar", [N("Gather", ["x", "i"], ["y"], axis=0)], [("x", rnd(4, 5))], ["y"], [("i", np.array(2, np.int64))])
model("embedding_gather", [N("Gather", ["w", "ids"], ["y"])], [("ids", rng.integers(0, 10, (2, 5)).astype(np.int64))], ["y"], [("w", rnd(10, 6))])
model("expand", [N("Expand", ["x", "s"], ["y"])], [("x", rnd(3, 1))], ["y"], [("s", np.array([2, 3, 4], np.int64))])
model("tile", [N("Tile", ["x", "r"], ["y"])], [("x", rnd(2, 3))], ["y"], [("r", np.array([2, 3], np.int64))])
model("pad_reflect", [N("Pad", ["x", "p"], ["y"], mode="reflect")], [("x", rnd(1, 2, 5, 5))], ["y"], [("p", np.array([0, 0, 1, 2, 0, 0, 2, 1], np.int64))])
model("resize_nearest", [N("Resize", ["x", "", "sc"], ["y"], mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")], [("x", rnd(1, 2, 4, 5))],
      ["y"], [("sc", np.array([1, 1, 2, 2], np.float32))])
model("resize_linear", [N("Resize", ["x", "", "sc"], ["y"], mode="linear")], [("x", rnd(1, 2, 4, 5))], ["y"], [("sc", np.array([1, 1, 2, 2], np.float32))])
model("argmax", [N("ArgMax", ["x"], ["y"], axis=1, keepdims=0)], [("x", rnd(4, 7))], ["y"])
model("topk", [N("TopK", ["x", "k"], ["v", "i"])], [("x", rnd(3, 10))], ["v", "i"], [("k", np.array([3], np.int64))])
# dynamic shape arithmetic folded on the host: Shape -> Gather -> Concat -> Reshape
model("shape_chain", [N("Shape", ["x"], ["s"]), N("Gather", ["s", "i0"], ["d0"]), N("Unsqueeze", ["d0", "z"], ["d0u"]),
                      N("Concat", ["d0u", "m1"], ["ns"], axis=0), N("Reshape", ["x", "ns"], ["y"])], [("x", rnd(3, 4, 5))], ["y"],
      [("i0", np.array(0, np.int64)), ("z", np.array([0], np.int64)), ("m1", np.array([-1], np.int64))])
model("constant_of_shape_range", [N("Shape", ["x"], ["s"]), N("ConstantOfShape", ["s"], ["c"], value=numpy_helper.from_array(np.array([0.5], np.float32))),
                                  N("Add", ["x", "c"], ["y"])], [("x", rnd(2, 3))], ["y"])


# ---- models exported from PyTorch
def torch_models():
    import torch

    class Cnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = torch.nn.Conv2d(3, 8, 3, padding=1)
            self.bn = torch.nn.BatchNorm2d(8)
            self.c2 = torch.nn.Conv2d(8, 16, 3, stride=2, padding=1)
            self.fc = torch.nn.Linear(16, 5)

        def forward(self, x):
            x = torch.relu(self.bn(self.c1(x)))
            x = torch.nn.functional.max_pool2d(x, 2)
            x = torch.relu(self.c2(x))
            return self.fc(x.mean((2, 3)))

    class Block(torch.nn.Module):
        def __init__(self, d=32, h=4):
            super().__init__()
            self.ln1 = torch.nn.LayerNorm(d)
            self.attn = torch.nn.MultiheadAttention(d, h, batch_first=True)
            self.ln2 = torch.nn.LayerNorm(d)
            self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.GELU(), torch.nn.Linear(4 * d, d))

        def forward(self, x):
            y = self.ln1(x)
            x = x + self.attn(y, y, y, need_weights=False)[0]
            return x + self.ff(self.ln2(x))

    torch.manual_seed(0)
    for name, m, x in (("torch_cnn", Cnn(), rnd(2, 3, 16, 16)), ("torch_transformer_block", Block(), rnd(2, 10, 32))):
        m.eval()
        for p in m.buffers():
            if p.dtype == torch.float32:
                p.data.uniform_(0.5, 1.5)
        path = os.path.join(OUT, name + ".onnx")
        torch.onnx.export(m, (torch.from_numpy(x),), path, input_names=["x"], output_names=["y"], opset_version=18, dynamo=True, external_data=False)
        cases.append((name, onnx.load(path), {"x": x}))


torch_models()

offset = 0
with open(os.path.join(OUT, "manifest.txt"), "w") as man, open(os.path.join(OUT, "data.bin"), "wb") as data:
    for name, m, feeds in cases:
        path = os.path.join(OUT, name + ".onnx")
        onnx.save(m, path)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        outs = sess.run(None, feeds)
        arrays = [(f"in:{k}", v) for k, v in feeds.items()] + [(f"out:{o.name}", r) for o, r in zip(sess.get_outputs(), outs)]
        for role, a in arrays:
            a = np.asarray(a)
            if a.dtype == np.bool_:
                a = a.astype(np.float32)
            kind = "i32" if a.dtype.kind in "iu" else "f32"
            b = a.astype("<i4" if kind == "i32" else "<f4").tobytes()
            man.write(f"{name} {role} {kind} {offset} {a.size} {' '.join(map(str, a.shape))}\n")
            data.write(b)
            offset += len(b)
print(len(cases), "models,", offset, "bytes")
