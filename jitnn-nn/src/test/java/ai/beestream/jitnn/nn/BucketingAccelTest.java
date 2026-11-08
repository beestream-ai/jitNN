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
package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;

import ai.beestream.jitnn.Buckets;
import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.JitnnRuntime;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.models.Bert;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.ResNet;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Models run on inputs padded to buckets give the results of the same models on the exact shapes
 * (GPT-2 causal, BERT with a ragged mask, ResNet-18 in evaluation mode, and an MLP's loss and
 * gradients), and a second request of another size in the same buckets compiles nothing.
 */
class BucketingAccelTest {

    @AfterAll
    static void release() {
        JitnnRuntime.reset();
    }

    private static void close(String what, float[] expected, float[] actual, float tolerance) {
        assertEquals(expected.length, actual.length, what + ": length");
        float scale = 1e-6f, worst = 0;
        for (float v : expected) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < expected.length; i++) {
            assertTrue(Float.isFinite(actual[i]), what + ": non-finite at " + i);
            worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
        }
        System.out.printf("[bucketing] %-28s max |diff| %.3g (scale %.3g)%n", what, worst, scale);
        assertTrue(worst <= tolerance * scale, what + ": max |diff| " + worst + " over " + tolerance + " x " + scale);
    }

    /** The plans compiled by {@code run}. */
    private static long compiles(Supplier<?> run) {
        long c = JitnnRuntime.planStats()[0];
        run.get();
        return JitnnRuntime.planStats()[0] - c;
    }

    private static Tensor tokens(Random r, int b, int t, int vocab) {
        int[] ids = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = r.nextInt(vocab);
        }
        return Tensor.of(ids, b, t);
    }

    /** Uniform values in [-1, 1) made on the host, so a new request never compiles a generator plan. */
    private static Tensor host(Random r, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = 2 * r.nextFloat() - 1;
        }
        return Tensor.of(v, dims);
    }

    @Test
    void bucketsRoundUp() {
        assertArrayEquals(new int[] { 8, 16, 32, 64 }, Buckets.powersOfTwo(5, 64).sizes());
        assertArrayEquals(new int[] { 1, 2, 4 }, Buckets.powersOfTwo(1, 3).sizes());
        assertArrayEquals(new int[] { 64, 128, 192 }, Buckets.multiplesOf(64, 150).sizes());
        assertEquals(16, Buckets.of(4, 16, 8).size(9));
        Tensor x = Tensor.of(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        Tensor p = Buckets.pad(x, 1, 5, -1);
        assertArrayEquals(new float[] { 1, 2, 3, -1, -1, 4, 5, 6, -1, -1 }, p.toFloatArray());
        assertArrayEquals(x.toFloatArray(), Buckets.cut(p, 1, 3).toFloatArray());
        Tensor rows = Buckets.pad(Tensor.of(new int[] { 7, 8 }, 2), 0, 4, 0);
        assertArrayEquals(new int[] { 7, 8, 0, 0 }, rows.toIntArray());
    }

    @Test
    void gpt2Causal() {
        int vocab = 500;
        GPT2 gpt2 = new GPT2(vocab, 128, 128, 4, 2, 1);
        Buckets batch = Buckets.powersOfTwo(1, 8), seq = Buckets.powersOfTwo(16, 128);
        Random r = new Random(1);
        Tensor idx = tokens(r, 3, 37, vocab);
        float[] exact = Grad.inference(() -> gpt2.forward(idx)).toFloatArray();
        float[] bucketed = Grad.inference(() -> Bucketing.causal(batch, seq, idx, 0, gpt2::forward)).toFloatArray();
        close("GPT-2 3x37 in 4x64", exact, bucketed, 1e-4f);
        Tensor other = tokens(r, 4, 50, vocab);
        assertEquals(0, compiles(() -> Grad.inference(() -> Bucketing.causal(batch, seq, other, 0, gpt2::forward)).toFloatArray()),
                "4x50 in the 4x64 bucket compiled again");
    }

    @Test
    void bertRaggedMask() {
        int vocab = 500, b = 3, t = 20;
        Bert bert = new Bert(vocab, 128, 128, 4, 2, 2);
        Buckets batch = Buckets.powersOfTwo(1, 8), seq = Buckets.powersOfTwo(16, 128);
        Random r = new Random(2);
        Tensor idx = tokens(r, b, t, vocab);
        float[] mask = new float[b * t];
        int[] lengths = { 20, 7, 13 };
        for (int row = 0; row < b; row++) {
            Arrays.fill(mask, row * t + lengths[row], (row + 1) * t, Bucketing.MASKED);
        }
        Tensor m = Tensor.of(mask, b, 1, 1, t);
        float[] exact = Grad.inference(() -> bert.encode(idx, m)).toFloatArray();
        float[] bucketed = Grad.inference(() -> Bucketing.masked(batch, seq, idx, m, 0, bert::encode)).toFloatArray();
        close("BERT 3x20 in 4x32", exact, bucketed, 1e-4f);
        Tensor other = tokens(r, 4, 31, vocab);
        assertEquals(0, compiles(() -> Grad.inference(() -> Bucketing.masked(batch, seq, other, null, 0, bert::encode)).toFloatArray()),
                "4x31 in the 4x32 bucket compiled again");
    }

    @Test
    void resnetRows() {
        Module resnet = ResNet.resnet18(3, 10, true, 3).train(false);
        Buckets batch = Buckets.powersOfTwo(1, 16);
        Tensor x = host(new Random(5), 5, 3, 32, 32);
        float[] exact = Grad.inference(() -> resnet.forward(x)).toFloatArray();
        float[] bucketed = Grad.inference(() -> Bucketing.rows(batch, x, resnet::forward)).toFloatArray();
        close("ResNet-18 5 in 8", exact, bucketed, 1e-4f);
        Tensor other = host(new Random(6), 7, 3, 32, 32);
        assertEquals(0, compiles(() -> Grad.inference(() -> Bucketing.rows(batch, other, resnet::forward)).toFloatArray()),
                "7 in the 8 bucket compiled again");
    }

    @Test
    void mlpTrainingLossAndGradients() {
        Sequential mlp = new Sequential(new Linear(64, 128, 4), new ReLU(), new Linear(128, 10, 5));
        Buckets batch = Buckets.powersOfTwo(1, 64);
        int n = 13;
        Tensor x = host(new Random(7), n, 64);
        int[] y = new Random(3).ints(n, 0, 10).toArray();
        Tensor labels = Tensor.of(y, n);

        Tensor exactLoss = F.crossEntropy(mlp.forward(x), labels);
        exactLoss.backward();
        float exact = exactLoss.item();
        List<Tensor> ps = mlp.parameters();
        float[][] exactGrads = new float[ps.size()][];
        for (int i = 0; i < ps.size(); i++) {
            exactGrads[i] = ps.get(i).grad().toFloatArray();
        }
        mlp.zeroGrad();

        Tensor loss = Bucketing.crossEntropy(mlp.forward(Buckets.pad(x, 0, batch.size(n), 0)), labels);
        loss.backward();
        close("MLP loss 13 in 16", new float[] { exact }, new float[] { loss.item() }, 1e-5f);
        for (int i = 0; i < ps.size(); i++) {
            close("MLP gradient " + i, exactGrads[i], ps.get(i).grad().toFloatArray(), 1e-4f);
        }
        mlp.zeroGrad();

        int m = 11;
        Tensor x2 = host(new Random(8), m, 64);
        Tensor y2 = Tensor.of(new Random(4).ints(m, 0, 10).toArray(), m);
        assertEquals(0, compiles(() -> {
            Tensor l = Bucketing.crossEntropy(mlp.forward(Buckets.pad(x2, 0, batch.size(m), 0)), y2);
            l.backward();
            return l.item();
        }), "11 in the 16 bucket compiled again");
        mlp.zeroGrad();
    }
}
