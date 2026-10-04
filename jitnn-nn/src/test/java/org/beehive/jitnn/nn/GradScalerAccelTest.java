package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.beehive.jitnn.nn.optim.GradScaler;
import org.beehive.jitnn.nn.optim.SGD;
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
}
