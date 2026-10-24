package org.beehive.jitnn.onnx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;
import org.beehive.jitnn.nn.models.Bert;
import org.beehive.jitnn.nn.models.GPT2;
import org.beehive.jitnn.nn.models.Llama;
import org.beehive.jitnn.nn.models.MobileNetV2;
import org.beehive.jitnn.nn.models.ResNet;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Models exported to ONNX and read back with jitNN's importer give the model's own outputs. With
 * {@code -Djitnn.onnx.exportDir=<dir>} the files, inputs and outputs are also written there for
 * scripts/onnx/check_export.py, which runs them in ONNX Runtime.
 */
class OnnxExportTest {

    @TempDir
    Path tmp;

    @BeforeAll
    static void reference() {
        Engine.setBackend(Engine.Backend.REFERENCE);
    }

    @Test
    void mlp() throws IOException {
        Sequential m = new Sequential(new Linear(12, 32, 1), new ReLU(), new Linear(32, 5, 2));
        check(tmp, "mlp", x -> m.forward(x[0]), Tensor.uniform(3, -1, 1, 4, 12));
    }

    @Test
    void resnet18() throws IOException {
        Module m = ResNet.resnet18(3, 10, true, 7).train(false);
        check(tmp, "resnet18_small", x -> m.forward(x[0]), Tensor.uniform(4, -1, 1, 2, 3, 32, 32));
    }

    @Test
    void mobilenetV2() throws IOException {
        Module m = MobileNetV2.mobilenetV2(3, 10, true, 0.0f, 9).train(false);
        check(tmp, "mobilenet_v2_small", x -> m.forward(x[0]), Tensor.uniform(5, -1, 1, 2, 3, 32, 32));
    }

    @Test
    void gpt2() throws IOException {
        GPT2 m = new GPT2(100, 16, 32, 4, 2, 11);
        check(tmp, "gpt2_tiny", x -> m.forward(x[0]), tokens(2, 16, 100));
    }

    @Test
    void llama() throws IOException {
        Llama m = new Llama(100, 32, 4, 2, 64, 10000.0f, 17);
        check(tmp, "llama_tiny", x -> m.forward(x[0]), tokens(2, 16, 100));
    }

    @Test
    void bertWithMask() throws IOException {
        Bert m = new Bert(100, 16, 32, 4, 2, 64, 2, true, 13);
        float[] mask = new float[2 * 16];
        for (int i = 12; i < 16; i++) {
            mask[16 + i] = -10000.0f;
        }
        check(tmp, "bert_tiny", x -> m.pooled(m.encode(x[0], x[1], x[2])), tokens(2, 16, 100), Tensor.of(new int[32], 2, 16), Tensor.of(mask, 2, 1, 1, 16));
    }

    static Tensor tokens(int b, int t, int vocab) {
        int[] ids = new int[b * t];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = (i * 37 + 5) % vocab;
        }
        return Tensor.of(ids, b, t);
    }

    static void check(Path tmp, String name, Function<Tensor[], Tensor> forward, Tensor... inputs) throws IOException {
        float[] want = forward.apply(inputs).toFloatArray();
        String[] names = new String[inputs.length];
        for (int i = 0; i < names.length; i++) {
            names[i] = "in" + i;
        }
        String out = System.getProperty("jitnn.onnx.exportDir", "");
        Path dir = out.isEmpty() ? tmp : Path.of(out);
        Files.createDirectories(dir);
        Path file = dir.resolve(name + ".onnx");
        OnnxExport.export(file, forward, inputs, names, "out");

        Map<String, Tensor> feeds = new HashMap<>();
        for (int i = 0; i < inputs.length; i++) {
            feeds.put(names[i], inputs[i]);
        }
        try (OnnxSession s = OnnxSession.load(file)) {
            float[] got = s.run(feeds).get("out").toFloatArray();
            assertEquals(want.length, got.length, name);
            double scale = 1e-6;
            double err = 0;
            for (int i = 0; i < want.length; i++) {
                scale = Math.max(scale, Math.abs(want[i]));
                err = Math.max(err, Math.abs(want[i] - got[i]));
            }
            System.out.printf("[export] %s: jitNN on the exported file vs the model, max error %.1e of %.3g%n", name, err, scale);
            assertTrue(err <= 1e-4 * scale, name + ": max error " + err + " of " + scale);
        }
        if (!out.isEmpty()) {
            for (int i = 0; i < inputs.length; i++) {
                Files.write(dir.resolve(name + ".in" + i + ".bin"), raw(inputs[i]));
            }
            Files.write(dir.resolve(name + ".out.f32"), raw(Tensor.of(want, want.length)));
        }
    }

    private static byte[] raw(Tensor t) {
        ByteBuffer b = ByteBuffer.allocate(4 * t.numel()).order(ByteOrder.LITTLE_ENDIAN);
        if (t.dtype() == org.beehive.jitnn.DType.I32) {
            for (int v : t.toIntArray()) {
                b.putInt(v);
            }
        } else {
            for (float v : t.toFloatArray()) {
                b.putFloat(v);
            }
        }
        return b.array();
    }
}
