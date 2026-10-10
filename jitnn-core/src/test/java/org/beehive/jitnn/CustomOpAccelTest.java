package org.beehive.jitnn;

import static org.beehive.jitnn.CustomOpFixtures.HYPOT;
import static org.beehive.jitnn.CustomOpFixtures.MIN_MAX;
import static org.beehive.jitnn.CustomOpFixtures.REPEAT_SCALED;
import static org.beehive.jitnn.CustomOpFixtures.SWISH;
import static org.beehive.jitnn.OpsAccelTest.same;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.beehive.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/** Custom operators on the TornadoVM device against their host references, and kernel printing. */
class CustomOpAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    void values() {
        same("custom swish", () -> SWISH.apply(Tensor.uniform(1, -4, 4, 1000, 37)));
        same("custom hypot", () -> HYPOT.apply(Tensor.uniform(2, -4, 4, 3001), Tensor.uniform(3, -4, 4, 3001)));
        same("custom min", () -> MIN_MAX.applyAll(null, null, Tensor.uniform(4, -1, 1, 700, 33))[0]);
        same("custom max", () -> MIN_MAX.applyAll(null, null, Tensor.uniform(4, -1, 1, 700, 33))[1]);
        same("custom repeat", () -> REPEAT_SCALED.apply(new int[] { 5 }, new float[] { -0.5f }, Tensor.uniform(5, -1, 1, 300)));
        // fused with built-in operators in one plan, and differentiated on the device
        same("custom in a graph", () -> SWISH.apply(Tensor.uniform(6, -1, 1, 64, 32).matmul(Tensor.uniform(7, -1, 1, 32, 16))).relu().add(1));
        same("custom gradient", () -> {
            Tensor x = Tensor.uniform(8, -2, 2, 50, 40).requiresGrad(true);
            Tensor w = Tensor.uniform(9, -1, 1, 50, 40).requiresGrad(true);
            HYPOT.apply(SWISH.apply(x), w).mul(Tensor.uniform(10, -1, 1, 50, 40)).sum().backward();
            return x.grad().add(w.grad());
        }, 1e-5, 1e-5);
    }

    @Test
    void printsKernels() {
        Engine.reset();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Engine.printKernels(new PrintStream(out, true));
        try {
            SWISH.apply(Tensor.uniform(11, -1, 1, 77, 3)).add(2).toFloatArray();
        } finally {
            Engine.printKernels(null);
        }
        String dump = out.toString();
        assertTrue(dump.contains("jitNN plan"), dump);
        assertTrue(dump.contains("custom:test.swish"), dump);
        // the generated source of the user's kernel
        assertTrue(dump.contains("swish"), dump);
        assertTrue(dump.split("swish", -1).length > 2, "no kernel source in:\n" + dump);
        System.out.println(dump);
    }
}
