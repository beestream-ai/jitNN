package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.optim.AdaDelta;
import org.beehive.jitnn.nn.optim.AdaGrad;
import org.beehive.jitnn.nn.optim.AdaMax;
import org.beehive.jitnn.nn.optim.Adam;
import org.beehive.jitnn.nn.optim.Lamb;
import org.beehive.jitnn.nn.optim.Lars;
import org.beehive.jitnn.nn.optim.NAdam;
import org.beehive.jitnn.nn.optim.RMSProp;
import org.beehive.jitnn.nn.optim.Optimizer;
import org.beehive.jitnn.nn.optim.SGD;
import org.junit.jupiter.api.DynamicTest;

/**
 * Three optimiser steps against torch.optim, from the optimiser cases of the core golden set
 * (jitnn-core/src/test/resources/golden). Gradients are injected as loss = sum(p * g).
 */
final class OptimGolden {

    private OptimGolden() {
    }

    static List<DynamicTest> tests(Engine.Backend backend) throws IOException {
        Path dir = Path.of("..", "jitnn-core", "src", "test", "resources", "golden");
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(dir.resolve("data.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        Map<String, Map<String, float[]>> cases = new HashMap<>();
        for (String line : Files.readAllLines(dir.resolve("manifest.txt"))) {
            String[] t = line.trim().split("\\s+");
            if (t.length < 5 || !t[0].startsWith("opt_")) {
                continue;
            }
            float[] v = new float[Integer.parseInt(t[4])];
            for (int k = 0; k < v.length; k++) {
                v[k] = data.getFloat(Integer.parseInt(t[3]) + 4 * k);
            }
            cases.computeIfAbsent(t[0].substring(4), k -> new HashMap<>()).put(t[1], v);
        }
        assertTrue(cases.size() >= 5, "optimiser golden cases found: " + cases.keySet());
        List<DynamicTest> tests = new ArrayList<>();
        for (var e : cases.entrySet()) {
            tests.add(dynamicTest(e.getKey(), () -> Engine.with(backend, () -> {
                run(e.getKey(), e.getValue());
                return null;
            })));
        }
        return tests;
    }

    private static void run(String name, Map<String, float[]> c) {
        float[] p0 = c.get("p0");
        Tensor p = Tensor.parameter(Tensor.of(p0, p0.length));
        List<Tensor> params = List.of(p);
        Optimizer opt = switch (name) {
            case "sgd" -> new SGD(params, 0.1f, 0, 0.01f, false);
            case "sgd_momentum" -> new SGD(params, 0.1f, 0.9f, 0.01f, false);
            case "sgd_nesterov" -> new SGD(params, 0.1f, 0.9f, 0, true);
            case "adam" -> new Adam(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, false);
            case "adamw" -> Adam.adamW(params, 0.01f, 0.05f);
            case "amsgrad" -> new Adam(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, false, true);
            case "adagrad" -> new AdaGrad(params, 0.1f, 0.01f, 0.01f, 0.1f, 1e-10f);
            case "rmsprop" -> new RMSProp(params, 0.01f, 0.9f, 1e-8f, 0.01f, 0, false);
            case "rmsprop_centered_momentum" -> new RMSProp(params, 0.01f, 0.9f, 1e-8f, 0, 0.8f, true);
            case "adadelta" -> new AdaDelta(params, 1.0f, 0.9f, 1e-6f, 0.01f);
            case "adamax" -> new AdaMax(params, 0.02f, 0.9f, 0.999f, 1e-8f, 0.01f);
            case "nadam" -> new NAdam(params, 0.01f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.004f);
            case "lamb" -> new Lamb(params, 0.01f, 0.9f, 0.999f, 1e-6f, 0.01f);
            case "lars" -> new Lars(params, 0.1f, 0.9f, 1e-4f, 0.001f, 1e-9f);
            default -> throw new IllegalArgumentException(name);
        };
        for (int k = 0; k < 3; k++) {
            float[] g = c.get("g" + k);
            p.mul(Tensor.of(g, g.length)).sum().backward();
            opt.step();
            opt.zeroGrad();
        }
        float[] expected = c.get("out");
        float[] actual = p.toFloatArray();
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], 1e-6 + 1e-5 * Math.abs(expected[i]), name + " element " + i);
        }
    }
}
