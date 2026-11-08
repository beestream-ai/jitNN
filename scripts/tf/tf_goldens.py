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
"""TensorFlow goldens for jitnn-tf: SavedModels and a frozen GraphDef, with TensorFlow's outputs.

Writes <out>/<case>/ (a SavedModel directory) or <out>/<case>.pb (frozen), plus <out>/manifest.txt
and <out>/data.bin: per case its kind, then its inputs and outputs (name, dtype, dims), their
values concatenated in data.bin as little-endian float32 / int32.

usage: python tf_goldens.py <out dir>     (TensorFlow 2.x with Keras 3, CPU)
"""
import os
import shutil
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(11)
keras.utils.set_random_seed(11)
manifest = []
blobs = []


def record(name, kind, inputs, outputs):
    manifest.append(f"case {name} {kind}")
    for n, a in inputs.items():
        dt = "i32" if a.dtype.kind in "iu" else "f32"
        manifest.append(f"input {n} {dt} " + " ".join(str(d) for d in a.shape))
        blobs.append(a.astype(np.int32 if dt == "i32" else np.float32).tobytes())
    for n, a in outputs.items():
        a = np.asarray(a)
        dt = "i32" if a.dtype.kind in "iub" else "f32"
        manifest.append(f"output {n} {dt} " + " ".join(str(d) for d in a.shape))
        blobs.append(a.astype(np.int32 if dt == "i32" else np.float32).tobytes())


def keras_case(name, model, inputs):
    path = os.path.join(out, name)
    shutil.rmtree(path, ignore_errors=True)
    model.export(path, verbose=False)
    loaded = tf.saved_model.load(path)
    fn = loaded.signatures["serving_default"]
    res = fn(**{k: tf.constant(v) for k, v in inputs.items()})
    record(name, "savedmodel", inputs, {k: v.numpy() for k, v in sorted(res.items())})


def input_names(model):
    return [t.name for t in model.inputs]


# ---- MLP
mlp = keras.Sequential([keras.Input((12,), name="features"), keras.layers.Dense(32, activation="relu"), keras.layers.Dense(16, activation="tanh"),
                        keras.layers.Dense(5, activation="softmax")])
keras_case("mlp", mlp, {"features": rng.standard_normal((3, 12)).astype(np.float32)})

# ---- CNN: same/valid convolutions, batch norm (inference), pooling, depthwise convolution
x = keras.Input((16, 16, 3), name="image")
h = keras.layers.Conv2D(8, 3, padding="same", activation="relu")(x)
h = keras.layers.BatchNormalization()(h)
h = keras.layers.MaxPooling2D(2)(h)
h = keras.layers.DepthwiseConv2D(3, padding="same", depth_multiplier=2)(h)
h = keras.layers.Conv2D(12, 3, strides=2, padding="valid", activation="relu6")(h)
h = keras.layers.AveragePooling2D(2, padding="same")(h)
h = keras.layers.GlobalAveragePooling2D()(h)
y = keras.layers.Dense(4)(h)
cnn = keras.Model(x, y)
# non-trivial batch-norm statistics
bn = [l for l in cnn.layers if isinstance(l, keras.layers.BatchNormalization)][0]
bn.set_weights([rng.uniform(0.5, 1.5, 8).astype(np.float32), rng.uniform(-0.5, 0.5, 8).astype(np.float32), rng.uniform(-0.2, 0.2, 8).astype(np.float32),
                rng.uniform(0.5, 2, 8).astype(np.float32)])
keras_case("cnn", cnn, {"image": rng.standard_normal((2, 16, 16, 3)).astype(np.float32)})

# ---- transformer block: embedding, multi-head attention (einsum), layer norm, GELU
ids = keras.Input((8,), dtype="int32", name="ids")
e = keras.layers.Embedding(50, 32)(ids)
a = keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)(e, e)
h = keras.layers.LayerNormalization()(e + a)
f = keras.layers.Dense(64, activation="gelu")(h)
f = keras.layers.Dense(32)(f)
h = keras.layers.LayerNormalization()(h + f)
y = keras.layers.Dense(3)(keras.layers.GlobalAveragePooling1D()(h))
block = keras.Model(ids, y)
keras_case("transformer", block, {"ids": rng.integers(0, 50, (2, 8)).astype(np.int32)})


# ---- TensorFlow operations through tf.function
class Ops(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec([2, 6, 8], tf.float32, name="x"), tf.TensorSpec([2, 3], tf.int32, name="idx")])
    def __call__(self, x, idx):
        img = tf.reshape(x[:, :, :6], [2, 6, 6, 1])
        return {
            "slice_basic": x[:, 1:5, ::2],
            "slice_shrink": x[1, :, -1],
            "slice_ellipsis": x[..., 2:4],
            "slice_newaxis": x[:, tf.newaxis, 0, :3],
            "slice_rev": x[:, ::-1, 1],
            "concat_pack": tf.concat([x, tf.stack([x[:, 0], x[:, 1]], axis=1)], axis=1),
            "unpack": tf.unstack(x, axis=1)[3],
            "split": tf.split(x, [2, 6], axis=2)[1],
            "transpose": tf.transpose(x, [2, 0, 1]),
            "reduce": tf.reduce_mean(x, axis=[1], keepdims=True) + tf.reduce_max(x, axis=2, keepdims=True),
            "reduce_sum": tf.reduce_sum(x, axis=[0, 2]),
            "argmax": tf.argmax(x, axis=2, output_type=tf.int32),
            "cumsum": tf.math.cumsum(x, axis=1),
            "matmul_t": tf.matmul(x[0], x[1], transpose_b=True),
            "batch_matmul_adj": tf.matmul(x, x, adjoint_a=True),
            "einsum": tf.einsum("bij,bkj->bik", x, x),
            "gather": tf.gather(x, idx[0], axis=1),
            "one_hot": tf.one_hot(idx, 5, on_value=2.0, off_value=-1.0),
            "tile": tf.tile(x[:, :2, :2], [1, 2, 3]),
            "pad": tf.pad(x, [[0, 0], [1, 2], [0, 1]]),
            "where": tf.where(x > 0, x, 0.5 * x),
            "math": tf.math.rsqrt(tf.abs(x) + 1) + tf.math.log1p(tf.abs(x)) + tf.square(x) - tf.math.squared_difference(x, 1.0) + tf.nn.relu6(x * 4),
            "softmax": tf.nn.softmax(x) + tf.nn.log_softmax(x),
            "cast_floordiv": tf.cast(tf.math.floordiv(idx, 2), tf.float32) + tf.cast(tf.math.floormod(idx, 3), tf.float32),
            "resize": tf.image.resize(img, [9, 9], method="bilinear"),
            "resize_nearest": tf.compat.v1.image.resize_nearest_neighbor(img, [12, 12]),
            "fill_range": tf.fill([2, 3], 1.5) + tf.cast(tf.range(3), tf.float32),
            "expand_squeeze": tf.squeeze(tf.expand_dims(x, 1), [1]) * 2,
        }


ops = Ops()
path = os.path.join(out, "ops")
shutil.rmtree(path, ignore_errors=True)
tf.saved_model.save(ops, path, signatures=ops.__call__)
xin = rng.standard_normal((2, 6, 8)).astype(np.float32)
idxin = rng.integers(0, 6, (2, 3)).astype(np.int32)
res = tf.saved_model.load(path).signatures["serving_default"](x=tf.constant(xin), idx=tf.constant(idxin))
record("ops", "savedmodel", {"x": xin, "idx": idxin}, {k: v.numpy() for k, v in sorted(res.items())})

# ---- frozen GraphDef (TensorFlow 1 style): the MLP with its variables folded into constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

conc = tf.function(lambda t: mlp(t)).get_concrete_function(tf.TensorSpec([3, 12], tf.float32, name="features"))
frozen = convert_variables_to_constants_v2(conc)
with open(os.path.join(out, "mlp_frozen.pb"), "wb") as f:
    f.write(frozen.graph.as_graph_def().SerializeToString())
fin = rng.standard_normal((3, 12)).astype(np.float32)
fout = frozen(tf.constant(fin))
fout = fout[0] if isinstance(fout, (list, tuple)) else fout
outs = [n.name for n in frozen.graph.as_graph_def().node if n.op == "Identity"]
record("mlp_frozen", "frozen", {"features": fin}, {outs[-1]: fout.numpy()})

with open(os.path.join(out, "manifest.txt"), "w") as f:
    f.write("\n".join(manifest) + "\n")
with open(os.path.join(out, "data.bin"), "wb") as f:
    for b in blobs:
        f.write(b)
print(len([m for m in manifest if m.startswith("case")]), "cases,", sum(len(b) for b in blobs), "bytes")
