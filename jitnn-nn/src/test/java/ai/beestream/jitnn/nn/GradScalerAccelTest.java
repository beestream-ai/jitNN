package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import ai.beestream.jitnn.nn.optim.GradScaler;
import ai.beestream.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/** Dynamic loss scaling and norm clipping inside the optimiser, on the host reference and on the device. */
class GradScalerAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    /** One scaled SGD step on loss = sum(p * c); returns [p after, overflow, scale after]. */
    private static float[] step(Tensor p, SGD opt, GradScaler scaler, float c) {
        Tensor loss = p.mul(c).sum();
        scaler.scale(loss).backward();
        scaler.step(opt);
        loss.item();
        boolean overflow = scaler.update();
        opt.zeroGrad();
        float[] v = p.toFloatArray();
        return new float[] { v[0], overflow ? 1 : 0, scaler.scale() };
    }

    private static void run(Backend backend) {
        Engine.with(backend, () -> {
            Tensor p = Tensor.parameter(Tensor.of(new float[] { 1.0f, 2.0f }, 2));
            SGD opt = new SGD(List.of(p), 0.5f);
            GradScaler scaler = new GradScaler(1024.0f, 2.0f, 0.5f, 2);
            // gradient 3 * 1024 is finite: p -= 0.5 * 3
            float[] r = step(p, opt, scaler, 3.0f);
            assertEquals(-0.5f, r[0], 1e-6f, backend + " clean step");
            assertEquals(0.0f, r[1], backend + " no overflow");
            // 1e36 * 1024 overflows FP32: the update is skipped and the scale halves
            r = step(p, opt, scaler, 1e36f);
            assertEquals(-0.5f, r[0], 1e-6f, backend + " skipped step");
            assertEquals(1.0f, r[1], backend + " overflow found");
            assertEquals(512.0f, r[2], backend + " scale after overflow");
            // two clean steps grow the scale back
            step(p, opt, scaler, 1.0f);
            r = step(p, opt, scaler, 1.0f);
            assertEquals(-1.5f, r[0], 1e-6f, backend + " after two clean steps");
            assertEquals(1024.0f, r[2], backend + " scale after growth");
            return null;
        });
    }

    @Test
    void referenceSkipsOverflow() {
        run(Backend.REFERENCE);
    }

    @Test
    void deviceSkipsOverflow() {
        run(Backend.TORNADO);
    }

    /** Gradients (3, 4): norm 5, clipped to 2.5 by a factor 0.5; with a loss scale the result is the same. */
    private static void clip(Backend backend, boolean scaled) {
        Engine.with(backend, () -> {
            Tensor p = Tensor.parameter(Tensor.of(new float[] { 1.0f, 2.0f }, 2));
            SGD opt = new SGD(List.of(p), 1.0f);
            GradScaler scaler = new GradScaler(1024.0f, 2.0f, 0.5f, 100);
            Tensor loss = p.mul(Tensor.of(new float[] { 3.0f, 4.0f }, 2)).sum();
            if (scaled) {
                scaler.scale(loss).backward();
                scaler.unscale(opt);
            } else {
                loss.backward();
            }
            Tensor norm = opt.clipGradNorm(2.5f);
            if (scaled) {
                scaler.step(opt);
            } else {
                opt.step();
            }
            assertEquals(5.0f, norm.item(), 1e-5f, backend + " norm");
            if (scaled) {
                assertFalse(scaler.update(), backend + " no overflow");
            }
            float[] v = p.toFloatArray();
            assertEquals(-0.5f, v[0], 1e-6f, backend + " clipped update 0");
            assertEquals(0.0f, v[1], 1e-6f, backend + " clipped update 1");
            return null;
        });
    }

    @Test
    void referenceClipsInsideTheUpdate() {
        clip(Backend.REFERENCE, false);
        clip(Backend.REFERENCE, true);
    }

    @Test
    void deviceClipsInsideTheUpdate() {
        clip(Backend.TORNADO, false);
        clip(Backend.TORNADO, true);
    }

    /** Three steps of an optimiser over 6 parameters of mixed sizes (two multi-tensor launches); returns the parameters. */
    private static float[][] multiTensor(Backend backend, String kind) {
        return Engine.with(backend, () -> {
            int[] sizes = { 1, 7, 300, 1000, 33, 2 };
            List<Tensor> params = new java.util.ArrayList<>();
            for (int i = 0; i < sizes.length; i++) {
                params.add(Tensor.parameter(Tensor.uniform(100 + i, -1.0f, 1.0f, sizes[i])));
            }
            ai.beestream.jitnn.nn.optim.Optimizer opt = switch (kind) {
                case "adamw" -> ai.beestream.jitnn.nn.optim.Adam.adamW(params, 0.01f, 0.05f);
                case "momentum" -> new SGD(params, 0.1f, 0.9f, 0.01f, true);
                default -> new SGD(params, 0.1f);
            };
            for (int step = 0; step < 3; step++) {
                Tensor loss = Tensor.scalar(0.0f);
                for (int i = 0; i < params.size(); i++) {
                    loss = loss.add(params.get(i).square().mul(i + 1.0f).sum());
                }
                loss.backward();
                opt.clipGradNorm(5.0f);
                opt.step();
                loss.item();
                opt.zeroGrad();
            }
            float[][] out = new float[params.size()][];
            for (int i = 0; i < out.length; i++) {
                out[i] = params.get(i).toFloatArray();
            }
            return out;
        });
    }

    @Test
    void multiTensorStepsMatchReference() {
        for (String kind : new String[] { "adamw", "momentum", "sgd" }) {
            float[][] expected = multiTensor(Backend.REFERENCE, kind);
            float[][] actual = multiTensor(Backend.TORNADO, kind);
            for (int i = 0; i < expected.length; i++) {
                for (int j = 0; j < expected[i].length; j++) {
                    assertEquals(expected[i][j], actual[i][j], 1e-5f + 1e-5f * Math.abs(expected[i][j]), kind + " parameter " + i + "[" + j + "]");
                }
            }
        }
    }
}
