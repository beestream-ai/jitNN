package ai.beestream.jitnn.onnx;

import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Stream;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * The inference benchmark's models from their ONNX files: ResNet-50 and BERT-base exported by PyTorch
 * (jitNN-benchmarking's scripts/inference.py), run by jitNN through the importer and compared with ONNX Runtime on the
 * same files and inputs (scripts/onnx/inference_onnx.py writes its outputs next to the models). Runs only
 * with {@code -Djitnn.onnx.inference=<work dir>}; {@code -Djitnn.onnx.inference.match=<substring>} picks files.
 */
class OnnxInferenceAccelTest {

    static final int WARMUP = 10;
    static final int RUNS = 50;
    static final int VOCAB = 30522;

    @AfterAll
    static void release() {
        Engine.reset();
    }

    static float[] images(int n) {
        float[] x = new float[n];
        for (int i = 0; i < n; i++) {
            x[i] = (float) ((i * 2654435761L) % 1000) / 500.0f - 1.0f;
        }
        return x;
    }

    static int[] tokens(int n) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = (int) ((i * 7919L + 13) % VOCAB);
        }
        return t;
    }

    static float[] read(Path f) throws Exception {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(f)).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[b.remaining() / 4];
        b.asFloatBuffer().get(out);
        return out;
    }

    static final double INT8_TOLERANCE = 5e-2;

    @Test
    void t1Models() throws Exception {
        String dir = System.getProperty("jitnn.onnx.inference", "");
        assumeTrue(!dir.isEmpty(), "set -Djitnn.onnx.inference=<dir with the inference benchmark's .onnx files>");
        String match = System.getProperty("jitnn.onnx.inference.match", "");
        Path work = Path.of(dir);
        String[] files;
        try (Stream<Path> s = Files.list(work)) {
            files = s.map(p -> p.getFileName().toString()).filter(f -> f.endsWith(".onnx") && f.contains(match)).sorted().toArray(String[]::new);
        }
        for (String f : files) {
            String name = f.substring(0, f.length() - 5);
            String[] parts = name.split("_");
            boolean fp16 = name.endsWith("fp16");
            long t0 = System.nanoTime();
            OnnxSession s = OnnxSession.load(work.resolve(f), fp16);
            double loadMs = (System.nanoTime() - t0) / 1e6;
            String input = s.inputNames().get(0);
            String output = s.outputNames().get(0);
            Tensor x;
            if (name.startsWith("resnet50")) {
                int b = Integer.parseInt(parts[1].substring(1));
                x = Tensor.of(images(b * 3 * 224 * 224), b, 3, 224, 224);
            } else {
                int b = Integer.parseInt(parts[1].substring(1));
                int t = Integer.parseInt(parts[2].substring(1));
                x = Tensor.of(tokens(b * t), b, t);
            }
            long c0 = System.nanoTime();
            float[] first = s.run(Map.of(input, x)).get(output).toFloatArray();
            double compile = (System.nanoTime() - c0) / 1e6;
            for (int i = 1; i < WARMUP; i++) {
                s.run(Map.of(input, x)).get(output).toFloatArray();
            }
            double[] ms = new double[RUNS];
            for (int i = 0; i < RUNS; i++) {
                long r0 = System.nanoTime();
                s.run(Map.of(input, x)).get(output).toFloatArray();
                ms[i] = (System.nanoTime() - r0) / 1e6;
            }
            Arrays.sort(ms);
            if (Boolean.getBoolean("jitnn.profile")) {
                // device time per operator over 10 calls (TornadoVM's profiler slows the calls themselves)
                Engine.kernelBreakdown();
                for (int i = 0; i < 10; i++) {
                    s.run(Map.of(input, x)).get(output).toFloatArray();
                }
                Map<String, long[]> perOp = new java.util.TreeMap<>();
                Engine.kernelBreakdown().forEach((k, v) -> {
                    long[] a = perOp.computeIfAbsent(k.replaceAll("#.*", ""), q -> new long[2]);
                    a[0] += v[0];
                    a[1] += v[1];
                });
                double total = perOp.values().stream().mapToLong(v -> v[1]).sum() / 1e7;
                System.out.printf("[t1-onnx] %s device time per call %.3f ms%n", name, total);
                perOp.entrySet().stream().sorted((a, b) -> Long.compare(b.getValue()[1], a.getValue()[1])).limit(12)
                        .forEach(e -> System.out.printf("[t1-onnx]   %-40s %8.3f ms %5d launches%n", e.getKey(), e.getValue()[1] / 1e7, e.getValue()[0] / 10));
            }
            String err = "no ONNX Runtime output";
            Path ref = work.resolve(name + ".ort.f32");
            double rel = -1;
            float[] want = null;
            String against = "ONNX Runtime";
            if (Files.exists(ref)) {
                want = read(ref);
            } else if (fp16) {
                // no ONNX Runtime output: the same file run in FP32 is the reference
                Engine.reset();
                want = OnnxSession.load(work.resolve(f), false).run(Map.of(input, x)).get(output).toFloatArray();
                against = "the FP32 session";
            }
            if (want != null) {
                assertTrue(want.length == first.length, name + ": " + first.length + " outputs, " + against + " has " + want.length);
                double scale = 0;
                double worst = 0;
                for (int i = 0; i < want.length; i++) {
                    scale = Math.max(scale, Math.abs(want[i]));
                    worst = Math.max(worst, Math.abs(want[i] - first[i]));
                }
                rel = worst / Math.max(scale, 1e-30);
                err = String.format("max err vs %s %.1e", against, rel);
            }
            System.out.printf("[t1-onnx] jitnn %-24s median %8.3f ms  p10 %8.3f  p90 %8.3f  (load %.0f ms, first call %.0f ms)  %s%n", name, ms[RUNS / 2], ms[RUNS / 10],
                    ms[RUNS * 9 / 10], loadMs, compile, err);
            if (rel >= 0) {
                // INT8 files: ONNX Runtime runs integer kernels, jitNN the float definition of the
                // operators, so an activation now and then lands one quantisation step apart
                double tol = name.contains("int8") ? INT8_TOLERANCE : fp16 ? 3e-2 : 1e-3;
                assertTrue(rel <= tol, name + ": " + err);
            }
            Engine.reset();
        }
    }
}
