package ai.beestream.jitnn.tf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

/** The TensorFlow goldens of scripts/tf/tf_goldens.py: each model's inputs and TensorFlow's outputs. */
final class TfGolden {

    static final Path DIR = Path.of("src/test/resources/tf");

    record Array(String name, boolean integer, int[] dims, float[] values) {
    }

    record Case(String name, String kind, List<Array> inputs, List<Array> outputs) {

        Path path() {
            return kind.equals("frozen") ? DIR.resolve(name + ".pb") : DIR.resolve(name);
        }
    }

    static List<Case> load() throws IOException {
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(DIR.resolve("data.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        List<Case> cases = new ArrayList<>();
        Case current = null;
        for (String line : Files.readAllLines(DIR.resolve("manifest.txt"))) {
            String[] f = line.trim().split(" ");
            if (f[0].equals("case")) {
                current = new Case(f[1], f[2], new ArrayList<>(), new ArrayList<>());
                cases.add(current);
                continue;
            }
            if (f.length < 3) {
                continue;
            }
            int[] dims = new int[f.length - 3];
            int n = 1;
            for (int i = 0; i < dims.length; i++) {
                dims[i] = Integer.parseInt(f[3 + i]);
                n *= dims[i];
            }
            boolean integer = f[2].equals("i32");
            float[] v = new float[n];
            for (int i = 0; i < n; i++) {
                v[i] = integer ? data.getInt() : data.getFloat();
            }
            (f[0].equals("input") ? current.inputs() : current.outputs()).add(new Array(f[1], integer, dims, v));
        }
        return cases;
    }

    static Map<String, Tensor> inputs(Case c) {
        Map<String, Tensor> in = new LinkedHashMap<>();
        for (Array a : c.inputs()) {
            if (a.integer()) {
                int[] iv = new int[a.values().length];
                for (int i = 0; i < iv.length; i++) {
                    iv[i] = (int) a.values()[i];
                }
                in.put(a.name(), Tensor.of(iv, a.dims()));
            } else {
                in.put(a.name(), Tensor.of(a.values(), a.dims()));
            }
        }
        return in;
    }

    /** Runs case {@code c} and checks every output against TensorFlow's. */
    static void check(Case c, double rtol, double atol) throws IOException {
        TfModel model = TfModel.load(c.path());
        Map<String, Tensor> out = model.run(inputs(c));
        for (Array expected : c.outputs()) {
            Tensor t = out.get(expected.name());
            if (t == null) {
                fail(c.name() + ": no output " + expected.name() + " among " + out.keySet());
            }
            float[] actual = (t.dtype() == DType.I32 ? t.to(DType.F32) : t).toFloatArray();
            assertEquals(expected.values().length, actual.length, c.name() + "/" + expected.name() + " size");
            for (int i = 0; i < actual.length; i++) {
                double e = expected.values()[i];
                if (Math.abs(actual[i] - e) > atol + rtol * Math.abs(e)) {
                    fail(c.name() + "/" + expected.name() + " [" + i + "]: TensorFlow " + e + ", jitNN " + actual[i]);
                }
            }
        }
    }
}
