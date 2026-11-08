package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import ai.beestream.jitnn.nn.models.GPT2;
import ai.beestream.jitnn.nn.models.ResNet;
import ai.beestream.jitnn.nn.optim.Adam;
import ai.beestream.jitnn.nn.optim.Optimizer;
import ai.beestream.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.Test;

/**
 * Whole training steps on the CPU backend against the reference: the losses of three steps and the
 * parameters after them, for a small ResNet-18 (convolutions, BatchNorm, pooling, SGD with momentum) and a small
 * GPT-2 (attention, LayerNorm, embeddings, AdamW). No operator may fall back to the reference.
 */
class CpuTrainingStepTest {

    private record Run(float[] losses, float[] params) {
    }

    private static Run train(Supplier<Module> model, java.util.function.Function<Module, Tensor> loss, java.util.function.Function<List<Tensor>, Optimizer> optimizer) {
        Module m = model.get();
        m.train(true);
        Optimizer opt = optimizer.apply(m.parameters());
        float[] losses = new float[3];
        for (int step = 0; step < 3; step++) {
            Tensor l = loss.apply(m);
            l.backward();
            opt.step();
            losses[step] = l.item();
        }
        List<Float> all = new ArrayList<>();
        for (Tensor p : m.parameters()) {
            for (float v : p.toFloatArray()) {
                all.add(v);
            }
        }
        float[] params = new float[all.size()];
        for (int i = 0; i < params.length; i++) {
            params[i] = all.get(i);
        }
        return new Run(losses, params);
    }

    private static void same(String what, Supplier<Run> run, double tol) {
        Run expected = Engine.with(Backend.REFERENCE, run::get);
        long[] before = fallbacks();
        Run actual = Engine.with(Backend.CPU, run::get);
        long[] after = fallbacks();
        close(expected.losses, actual.losses, tol, tol, what + " losses");
        close(expected.params, actual.params, tol, tol, what + " parameters");
        assertEquals(before[0], after[0], what + ": operators that fell back to the reference " + Engine.cpuStats().keySet());
    }

    private static void close(float[] expected, float[] actual, double rtol, double atol, String what) {
        assertEquals(expected.length, actual.length, what + ": length");
        for (int i = 0; i < expected.length; i++) {
            double tol = atol + rtol * Math.abs(expected[i]);
            if (Math.abs(expected[i] - actual[i]) > tol) {
                org.junit.jupiter.api.Assertions.fail(what + ": element " + i + " expected " + expected[i] + " got " + actual[i]);
            }
        }
    }

    private static long[] fallbacks() {
        long n = 0;
        for (long[] v : Engine.cpuStats().values()) {
            n += v[2];
        }
        return new long[] { n };
    }

    @Test
    void resnet18() {
        // 32 x 32 images: the last stage's 2 x 2 maps give BatchNorm 16 values per channel (at 16 x 16, a
        // single pixel per image leaves 4, and the normalisation amplifies FP32 rounding)
        same("ResNet-18, 4 images of 32 x 32", () -> train(() -> ResNet.resnet18(3, 10, true, 1),
                m -> F.crossEntropy(m.forward(Tensor.uniform(5, -1, 1, 4, 3, 32, 32)), Tensor.of(new int[] { 1, 7, 3, 0 }, 4)),
                // SGD: an update linear in the gradient (Adam's is near +-lr wherever a gradient is near
                // zero, so rounding that flips its sign moves a parameter by up to lr)
                p -> new SGD(p, 0.01f, 0.9f, 0f, false)), 2e-4);
    }

    @Test
    void gpt2() {
        int b = 2, t = 16, vocab = 50;
        int[] tokens = new int[b * t];
        int[] targets = new int[b * t];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = (i * 7) % vocab;
            targets[i] = (i * 7 + 3) % vocab;
        }
        same("GPT-2, 2 layers of 32", () -> train(() -> new GPT2(vocab, t, 32, 4, 2, 1),
                m -> F.crossEntropy(m.forward(Tensor.of(tokens, b, t)).reshape(b * t, vocab), Tensor.of(targets, b * t)),
                p -> new Adam(p, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, true)), 2e-4);
    }
}
