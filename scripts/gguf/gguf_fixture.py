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
"""Writes the GGUF test file for jitnn-gguf with llama.cpp's gguf package.

    python scripts/gguf/gguf_fixture.py jitnn-gguf/src/test/resources/gguf/fixture.gguf

It holds a metadata value of every GGUF type, and for every tensor type jitNN loads a weight
"w.<type>" of shape [3, 512] made of random valid blocks, next to "expect.<type>", its values
dequantised by the gguf package, in F32. Also: a 1-D and a 3-D F32 tensor with known values (axis
order), an I32 tensor, and a Q8_K tensor (a type jitNN does not load).
"""
import sys

import numpy as np
import gguf

rng = np.random.default_rng(11)
Q = gguf.GGMLQuantizationType

# block formats: values per block, bytes per block, byte offsets of the FP16 scales
BLOCKS = {
    "Q8_0": (32, 34, [0]),
    "Q4_0": (32, 18, [0]),
    "Q4_1": (32, 20, [0, 2]),
    "Q4_K": (256, 144, [0, 2]),
    "Q5_K": (256, 176, [0, 2]),
    "Q6_K": (256, 210, [208]),
    "Q5_0": (32, 22, [0]),
    "Q5_1": (32, 24, [0, 2]),
    "Q2_K": (256, 84, [80, 82]),
    "Q3_K": (256, 110, [108]),
    "Q8_K": (256, 292, []),
}


def blocks(name, rows, cols):
    qk, nb, scales = BLOCKS[name]
    raw = rng.integers(0, 256, size=(rows * cols // qk, nb), dtype=np.uint8)
    for off in scales:
        raw[:, off:off + 2] = rng.uniform(0.002, 0.03, size=(raw.shape[0], 1)).astype(np.float16).view(np.uint8)
    return raw.reshape(rows, cols // qk * nb)


def main(out):
    w = gguf.GGUFWriter(out, "llama")
    w.add_uint8("test.u8", 250)
    w.add_int8("test.i8", -100)
    w.add_uint16("test.u16", 60000)
    w.add_int16("test.i16", -30000)
    w.add_uint32("test.u32", 3_000_000_000)
    w.add_int32("test.i32", -2_000_000_000)
    w.add_float32("test.f32", 1.5)
    w.add_bool("test.bool", True)
    w.add_string("test.string", "héllo, wörld")
    w.add_uint64("test.u64", 2**40 + 3)
    w.add_int64("test.i64", -(2**40))
    w.add_float64("test.f64", 2.25)
    w.add_array("test.ints", [1, -2, 3])
    w.add_array("test.floats", [0.5, -1.25])
    w.add_array("test.strings", ["a", "βγ", ""])
    w.add_array("test.bools", [True, False])
    w.add_array("test.nested", [[1, 2], [3]])
    w.add_key_value("test.u32s", [7, 4_000_000_000], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.UINT32)
    w.add_key_value("test.f64s", [0.125, -3.5], gguf.GGUFValueType.ARRAY, sub_type=gguf.GGUFValueType.FLOAT64)
    w.add_block_count(2)
    w.add_embedding_length(512)

    rows, cols = 3, 512
    dense = rng.uniform(-2, 2, size=(rows, cols)).astype(np.float32)
    w.add_tensor("w.F32", dense)
    w.add_tensor("expect.F32", dense)
    half = dense.astype(np.float16)
    w.add_tensor("w.F16", half)
    w.add_tensor("expect.F16", half.astype(np.float32))
    bf = (dense.view(np.uint32) >> 16).astype(np.uint16)
    w.add_tensor("w.BF16", bf.view(np.uint8).reshape(rows, cols * 2), raw_dtype=Q.BF16)
    w.add_tensor("expect.BF16", (bf.astype(np.uint32) << 16).view(np.float32))
    for name in ["Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K", "Q5_0", "Q5_1", "Q2_K", "Q3_K"]:
        raw = blocks(name, rows, cols)
        w.add_tensor(f"w.{name}", raw, raw_dtype=getattr(Q, name))
        w.add_tensor(f"expect.{name}", gguf.quants.dequantize(raw, getattr(Q, name)).astype(np.float32))

    # a type jitNN does not load
    w.add_tensor("w.Q8_K", blocks("Q8_K", rows, cols), raw_dtype=Q.Q8_K)
    w.add_tensor("axes.1d", np.arange(5, dtype=np.float32))
    w.add_tensor("axes.3d", np.arange(24, dtype=np.float32).reshape(2, 3, 4))
    w.add_tensor("ints.I32", np.array([[1, -2, 3], [4, 5, -6]], dtype=np.int32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main(sys.argv[1])
