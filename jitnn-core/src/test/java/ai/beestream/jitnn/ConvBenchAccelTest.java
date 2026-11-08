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

import java.util.function.Supplier;

import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.ops.Conv2dGrad;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Convolution, normalisation and pooling kernels on ResNet/MobileNet layer shapes (batch 64): convolution forward, input gradient
 * and weight gradient in GFLOP/s, BatchNorm and pooling in GB/s. Printed, not asserted. Run with
 * {@code -Dtest=ConvBenchAccelTest -Djitnn.test.profile=true}.
 */
class ConvBenchAccelTest {

    private static final int REPS = 20;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    static Tensor persistent(int... dims) {
        return Tensor.persistent(Tensor.uniform(3, -1, 1, dims));
    }

    static void time(String name, double work, String unit, Supplier<Tensor> op) {
        for (int i = 0; i < 3; i++) {
            op.get();
            Engine.sync();
        }
        long[] kernel = new long[REPS];
        long t0 = System.nanoTime();
        for (int i = 0; i < REPS; i++) {
            op.get();
            Engine.sync();
            kernel[i] = Engine.lastProfile()[0];
        }
        double wall = (System.nanoTime() - t0) / 1e9 / REPS;
        java.util.Arrays.sort(kernel);
        double k = kernel[REPS / 2] / 1e9;
        System.out.printf("[conv] %-34s %8.3f ms %8.1f %s   kernel %8.3f ms %8.1f %s%n", name, wall * 1e3, work / wall / 1e9, unit, k * 1e3, k > 0 ? work / k / 1e9 : 0, unit);
    }

    static void conv(String name, int n, int cin, int hw, int cout, int k, int stride, int groups) {
        int pad = k / 2;
        int ohw = (hw + 2 * pad - k) / stride + 1;
        double flops = 2.0 * n * cout * ohw * ohw * (cin / groups) * k * k;
        Tensor x = persistent(n, cin, hw, hw);
        Tensor w = persistent(cout, cin / groups, k, k);
        Tensor dy = persistent(n, cout, ohw, ohw);
        time(name + " fwd", flops, "GFLOP/s", () -> F.conv2d(x, w, null, stride, pad, 1, groups));
        time(name + " dgrad", flops, "GFLOP/s", () -> Conv2dGrad.input(dy, w, x.shape(), stride, pad, 1, groups));
        time(name + " wgrad", flops, "GFLOP/s", () -> Conv2dGrad.weight(dy, x, w.shape(), stride, pad, 1, groups));
    }

    @Test
    void convolutionKernels() {
        conv("conv3x3 64->64 28x28", 64, 64, 28, 64, 3, 1, 1);
        conv("conv3x3 128->128 14x14", 64, 128, 14, 128, 3, 1, 1);
        conv("conv3x3 256->256 7x7", 64, 256, 7, 256, 3, 1, 1);
        conv("conv1x1 96->576 7x7", 64, 96, 7, 576, 1, 1, 1);
        conv("depthwise3x3 576 7x7", 64, 576, 7, 576, 3, 1, 576);
        Tensor x = persistent(64, 64, 28, 28);
        Tensor rm = Tensor.persistent(Tensor.zeros(64));
        Tensor rv = Tensor.persistent(Tensor.ones(64));
        Tensor g = persistent(64);
        Tensor b = persistent(64);
        double bytes = 4.0 * x.numel();
        time("batchnorm train 64x64x28x28", 3 * bytes, "GB/s", () -> F.batchNorm(x, rm, rv, g, b, true, 0.1f, 1e-5f));
        time("maxpool 3x3/2 64x64x28x28", 1.25 * bytes, "GB/s", () -> F.maxPool2d(x, 3, 2, 1));
    }
}
