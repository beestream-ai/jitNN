/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.onnx;

import java.util.Arrays;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

/**
 * A value flowing through an ONNX graph: either a jitNN tensor, or a small integer tensor known
 * on the host (shapes, axes, indices built from constants), which shape arithmetic folds without
 * touching the device.
 */
final class Value {

    final Tensor tensor;
    final long[] ints;
    final long[] dims;
    /** The ONNX element type this value stands for. */
    final int type;
    /** For small float constants (initializers, Constant nodes): the values, readable without a device sync. */
    float[] floats;

    private Value(Tensor tensor, long[] ints, long[] dims, int type) {
        this.tensor = tensor;
        this.ints = ints;
        this.dims = dims;
        this.type = type;
    }

    static Value constant(Tensor t, float[] values) {
        Value v = of(t, OnnxModel.FLOAT);
        v.floats = values.length <= 64 ? values : null;
        return v;
    }

    /** The values when known on the host (host integers or small float constants), else null. */
    float[] hostFloats() {
        if (floats != null) {
            return floats;
        }
        if (ints != null) {
            float[] f = new float[ints.length];
            for (int i = 0; i < f.length; i++) {
                f[i] = ints[i];
            }
            return f;
        }
        return null;
    }

    static Value of(Tensor t, int type) {
        long[] d = new long[t.rank()];
        for (int i = 0; i < d.length; i++) {
            d[i] = t.dim(i);
        }
        return new Value(t, null, d, type);
    }

    static Value of(Tensor t) {
        return of(t, t.dtype() == DType.I32 ? OnnxModel.INT64 : OnnxModel.FLOAT);
    }

    static Value host(long[] ints, long[] dims) {
        return new Value(null, ints, dims, OnnxModel.INT64);
    }

    /** Host integers of ONNX element type {@code type} (INT8 and UINT8 constants keep theirs: quantisation reads it). */
    static Value host(long[] ints, long[] dims, int type) {
        return new Value(null, ints, dims, type);
    }

    static Value scalar(long v) {
        return host(new long[] { v }, new long[0]);
    }

    boolean isHost() {
        return ints != null;
    }

    int rank() {
        return dims.length;
    }

    int[] intDims() {
        int[] d = new int[dims.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = (int) dims[i];
        }
        return d;
    }

    int numel() {
        return OnnxModel.elements("value", dims);
    }

    /** As a jitNN tensor: host integers become an INT32 tensor, or FP32 when {@code asFloat}. */
    Tensor tensor(boolean asFloat) {
        if (tensor != null) {
            return tensor;
        }
        int[] d = intDims();
        if (asFloat) {
            float[] f = new float[ints.length];
            for (int i = 0; i < f.length; i++) {
                f[i] = ints[i];
            }
            return Tensor.of(f, d.length == 0 ? new int[] { 1 } : d).reshape(d);
        }
        int[] v = new int[ints.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = Math.toIntExact(ints[i]);
        }
        return Tensor.of(v, d.length == 0 ? new int[] { 1 } : d).reshape(d);
    }

    /** The host integers, reading a device tensor back only when it has to (it ends the current trace). */
    long[] longs() {
        if (ints != null) {
            return ints;
        }
        float[] f = tensor.toFloatArray();
        long[] out = new long[f.length];
        for (int i = 0; i < f.length; i++) {
            out[i] = (long) f[i];
        }
        return out;
    }

    @Override
    public String toString() {
        return (isHost() ? "host" + Arrays.toString(ints) : "tensor") + Arrays.toString(dims);
    }
}
