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
package ai.beestream.jitnn;

import ai.beestream.jitnn.engine.Emitter;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Custom operators written as a user would: KernelContext kernels, host references and gradients. */
final class CustomOpFixtures {

    private CustomOpFixtures() {
    }

    public static void swish(KernelContext kc, FloatArray x, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float v = x.get(i);
            y.set(i, v / (1.0f + TornadoMath.exp(-v)));
        }
    }

    public static void hypotenuse(KernelContext kc, FloatArray a, FloatArray b, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float p = a.get(i);
            float q = b.get(i);
            y.set(i, TornadoMath.sqrt(p * p + q * q));
        }
    }

    /** One thread per row: the row's minimum and maximum. */
    public static void minMax(KernelContext kc, FloatArray x, FloatArray min, FloatArray max, int rows, int cols) {
        int r = kc.globalIdx;
        if (r < rows) {
            float lo = x.get(r * cols);
            float hi = lo;
            for (int c = 1; c < cols; c++) {
                float v = x.get(r * cols + c);
                lo = TornadoMath.min(lo, v);
                hi = TornadoMath.max(hi, v);
            }
            min.set(r, lo);
            max.set(r, hi);
        }
    }

    /** y = x repeated k times, scaled by s. */
    public static void repeatScaled(KernelContext kc, FloatArray x, FloatArray y, int n, int total, float s) {
        int i = kc.globalIdx;
        if (i < total) {
            y.set(i, s * x.get(i % n));
        }
    }

    /** Named like a CUDA built-in: rejected. */
    public static void exp(KernelContext kc, FloatArray x, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            y.set(i, TornadoMath.exp(x.get(i)));
        }
    }

    // d/dx x s(x) = y + s(x) (1 - y)
    static final CustomOp SWISH = CustomOp.unary("test.swish", CustomOpFixtures::swish, v -> v / (1 + Math.exp(-v)))
            .withGradient((x, y, gy) -> gy.mul(y.add(x.sigmoid().mul(y.neg().add(1)))));

    static final CustomOp HYPOT = CustomOp.binary("test.hypot", CustomOpFixtures::hypotenuse, Math::hypot)
            .withGradient((a, b, y, gy) -> new Tensor[] { gy.mul(a.div(y)), gy.mul(b.div(y)) });

    static final CustomOp MIN_MAX = CustomOp.builder("test.minMax")
            .inputs(DType.F32)
            .output((in, a) -> Shape.of(in[0].shape().dim(0)), DType.F32)
            .output((in, a) -> Shape.of(in[0].shape().dim(0)), DType.F32)
            .launch((e, n) -> {
                FloatArray x = e.f32(n.in(0));
                FloatArray min = e.f32(n.out(0));
                FloatArray max = e.f32(n.out(1));
                int rows = n.in(0).shape().dim(0);
                int cols = n.in(0).shape().dim(1);
                e.task(Emitter.grid1D(rows), (g, id) -> g.task(id, CustomOpFixtures::minMax, e.context(), x, min, max, rows, cols));
            })
            .reference(n -> {
                float[] x = n.in(0).toFloatArray();
                int rows = n.in(0).shape().dim(0);
                int cols = n.in(0).shape().dim(1);
                FloatArray min = (FloatArray) n.out(0).storage().host();
                FloatArray max = (FloatArray) n.out(1).storage().host();
                for (int r = 0; r < rows; r++) {
                    float lo = Float.POSITIVE_INFINITY;
                    float hi = Float.NEGATIVE_INFINITY;
                    for (int c = 0; c < cols; c++) {
                        lo = Math.min(lo, x[r * cols + c]);
                        hi = Math.max(hi, x[r * cols + c]);
                    }
                    min.set(r, lo);
                    max.set(r, hi);
                }
            })
            .build();

    /** iattr = [k], fattr = [s]. */
    static final CustomOp REPEAT_SCALED = CustomOp.builder("test.repeatScaled")
            .inputs(DType.F32)
            .output((in, a) -> Shape.of(in[0].numel() * a[0]), DType.F32)
            .launch((e, n) -> {
                FloatArray x = e.f32(n.in(0));
                FloatArray y = e.f32(n.out(0));
                int len = n.in(0).numel();
                int total = n.out(0).numel();
                float s = n.fattr[0];
                e.task(Emitter.grid1D(total), (g, id) -> g.task(id, CustomOpFixtures::repeatScaled, e.context(), x, y, len, total, s));
            })
            .reference(n -> {
                float[] x = n.in(0).toFloatArray();
                FloatArray y = (FloatArray) n.out(0).storage().host();
                for (int i = 0; i < n.out(0).numel(); i++) {
                    y.set(i, n.fattr[0] * x[i % x.length]);
                }
            })
            .backward((n, gy) -> new Tensor[] { gy.reshape(n.iattr[0], n.in(0).numel()).sum(0, false).mul(n.fattr[0]).reshape(n.in(0).shape().dims()) })
            .build();
}
