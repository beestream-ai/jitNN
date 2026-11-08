package ai.beestream.jitnn.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.DynamicTest.dynamicTest;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.DynamicTest;

/**
 * The ONNX goldens (scripts/onnx/onnx_goldens.py, in src/test/resources/onnx): each model run by
 * jitNN must give ONNX Runtime's outputs, floats within 2e-4 of the output's scale and integers exactly.
 */
final class OnnxGolden {

    record Arr(int[] dims, float[] f, int[] i) {
    }

    private OnnxGolden() {
    }

    static Path dir() {
        return Path.of("src", "test", "resources", "onnx");
    }

    static Map<String, Map<String, Arr>> load() throws IOException {
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(dir().resolve("data.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        Map<String, Map<String, Arr>> cases = new LinkedHashMap<>();
        for (String line : Files.readAllLines(dir().resolve("manifest.txt"))) {
            String[] t = line.trim().split("\\s+");
            int offset = Integer.parseInt(t[3]);
            int count = Integer.parseInt(t[4]);
            int[] dims = new int[t.length - 5];
            for (int k = 0; k < dims.length; k++) {
                dims[k] = Integer.parseInt(t[5 + k]);
            }
            Arr a;
            if (t[2].equals("i32")) {
                int[] v = new int[count];
                for (int k = 0; k < count; k++) {
                    v[k] = data.getInt(offset + 4 * k);
                }
                a = new Arr(dims, null, v);
            } else {
                float[] v = new float[count];
                for (int k = 0; k < count; k++) {
                    v[k] = data.getFloat(offset + 4 * k);
                }
                a = new Arr(dims, v, null);
            }
            cases.computeIfAbsent(t[0], k -> new LinkedHashMap<>()).put(t[1], a);
        }
        return cases;
    }

    static List<DynamicTest> tests(Engine.Backend backend) throws IOException {
        List<DynamicTest> tests = new ArrayList<>();
        for (var c : load().entrySet()) {
            tests.add(dynamicTest(c.getKey(), () -> Engine.with(backend, () -> {
                run(c.getKey(), c.getValue());
                return null;
            })));
        }
        return tests;
    }

    static void run(String name, Map<String, Arr> arrays) {
        OnnxSession s;
        try {
            s = OnnxSession.load(dir().resolve(name + ".onnx"));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        Map<String, Tensor> inputs = new HashMap<>();
        for (var e : arrays.entrySet()) {
            if (e.getKey().startsWith("in:")) {
                Arr a = e.getValue();
                inputs.put(e.getKey().substring(3), a.f != null ? Tensor.of(a.f, a.dims) : Tensor.of(a.i, a.dims));
            }
        }
        Map<String, Tensor> out = s.run(inputs);
        for (var e : arrays.entrySet()) {
            if (!e.getKey().startsWith("out:")) {
                continue;
            }
            String o = e.getKey().substring(4);
            Tensor y = out.get(o);
            assertTrue(y != null, name + ": no output " + o);
            Arr want = e.getValue();
            if (want.i != null) {
                assertArrayEquals(want.i, y.toIntArray(), name + " output " + o);
                continue;
            }
            float[] got = y.toFloatArray();
            assertTrue(got.length == want.f.length, name + " output " + o + ": " + got.length + " values, expected " + want.f.length);
            double scale = 1;
            for (float v : want.f) {
                scale = Math.max(scale, Math.abs(v));
            }
            double worst = 0;
            int at = 0;
            for (int i = 0; i < got.length; i++) {
                double d = Math.abs(got[i] - want.f[i]);
                if (!(d <= worst)) {
                    worst = d;
                    at = i;
                }
            }
            assertTrue(worst <= 2e-4 * scale, name + " output " + o + ": element " + at + " expected " + want.f[at] + " got " + got[at]);
        }
    }
}
