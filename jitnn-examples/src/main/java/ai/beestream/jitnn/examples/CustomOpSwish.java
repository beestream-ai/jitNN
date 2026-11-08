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
package ai.beestream.jitnn.examples;

import ai.beestream.jitnn.CustomOp;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.data.DataLoader;
import ai.beestream.jitnn.data.Mnist;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.Sequential;
import ai.beestream.jitnn.nn.optim.Adam;

import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.math.TornadoMath;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/**
 * Your own GPU operator in plain Java: swish, x * sigmoid(x), written as a TornadoVM
 * {@code KernelContext} kernel with a host reference and a gradient. jitNN records it like a
 * built-in operator, compiles it into the same plan as its neighbours and differentiates it. The
 * example checks it against jitNN's built-in {@code silu}, then trains an MNIST network with it.
 *
 * <pre>run.sh CustomOpSwish [epochs]</pre>
 */
public final class CustomOpSwish {

    /** One thread per element, bounds-checked: the grid is rounded up to whole thread blocks. */
    public static void swishKernel(KernelContext kc, FloatArray x, FloatArray y, int n) {
        int i = kc.globalIdx;
        if (i < n) {
            float v = x.get(i);
            y.set(i, v / (1.0f + TornadoMath.exp(-v)));
        }
    }

    static final CustomOp SWISH = CustomOp.unary("swish", CustomOpSwish::swishKernel, v -> v / (1 + Math.exp(-v)))
            // d/dx x sigmoid(x) = y + sigmoid(x) (1 - y)
            .withGradient((x, y, gy) -> gy.mul(y.add(x.sigmoid().mul(y.neg().add(1)))));

    /** The operator as a layer. */
    static final class Swish extends Module {
        @Override
        public Tensor forward(Tensor x) {
            return SWISH.apply(x);
        }
    }

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 3);

        // forward and gradient against the built-in silu
        Tensor x = Tensor.uniform(1, -6, 6, 1 << 20).requiresGrad(true);
        Tensor mine = SWISH.apply(x);
        mine.sum().backward();
        float[] myValues = mine.toFloatArray();
        float[] myGrad = x.grad().toFloatArray();
        Tensor x2 = Tensor.uniform(1, -6, 6, 1 << 20).requiresGrad(true);
        Tensor builtIn = x2.silu();
        builtIn.sum().backward();
        System.out.printf("custom swish vs built-in silu on %,d values: largest difference %.2e, gradient %.2e%n", myValues.length,
                maxDiff(myValues, builtIn.toFloatArray()), maxDiff(myGrad, x2.grad().toFloatArray()));

        // an MNIST network with the custom activation
        int batch = 64;
        Mnist train = Mnist.load(Examples.mnist(), true);
        Mnist test = Mnist.load(Examples.mnist(), false);
        Module model = new Sequential(new Linear(784, 256, 1), new Swish(), new Linear(256, 128, 2), new Swish(), new Linear(128, 10, 3));
        Adam opt = new Adam(model.parameters(), 1e-3f);
        try (DataLoader loader = DataLoader.builder(train, batch).shuffle(42).dropLast(true).build()) {
            for (int epoch = 0; epoch < epochs; epoch++) {
                long t0 = System.nanoTime();
                float sum = 0;
                for (DataLoader.Batch b : loader.epoch(epoch)) {
                    Tensor loss = F.crossEntropy(model.forward(b.x().reshape(batch, 784)), b.y());
                    loss.backward();
                    opt.step();
                    sum += loss.item();
                    opt.zeroGrad();
                }
                System.out.printf("epoch %d: loss %.4f, test accuracy %.2f%%, %.1f s%n", epoch + 1, sum / loader.batches(), 100 * accuracy(model, test), Examples.seconds(t0));
            }
        }
    }

    static float accuracy(Module model, Mnist test) {
        int n = 1000;
        float correct = 0;
        try (DataLoader loader = DataLoader.builder(test, n).build()) {
            for (DataLoader.Batch b : loader.epoch(0)) {
                correct += F.accuracy(Grad.inference(() -> model.forward(b.x().reshape(n, 784))), b.y()) * n;
            }
        }
        return correct / test.count();
    }

    static float maxDiff(float[] a, float[] b) {
        float m = 0;
        for (int i = 0; i < a.length; i++) {
            m = Math.max(m, Math.abs(a[i] - b[i]));
        }
        return m;
    }
}
