package bench;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Supplier;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;

/**
 * DJL inference timing with the protocol of scripts/bench/t1.py: host input in, host output out,
 * 10 warm-up calls, then the median of 50. The inputs are the deterministic ones t1.py and jitNN's
 * InferenceBenchAccelTest build.
 *
 * <p>usage: Bench &lt;OnnxRuntime|PyTorch&gt; &lt;model file&gt; &lt;resnet50|bert&gt; &lt;batch&gt; &lt;seq&gt; &lt;fp32|fp16&gt; &lt;label&gt;
 */
public final class Bench {

    static final int WARMUP = 10;
    static final int RUNS = 50;
    static final int VOCAB = 30522;

    public static void main(String[] args) throws Exception {
        // --with-native: let jitNN call cuBLAS, cuDNN, ... (or MLX) where they are available
        if (java.util.Arrays.asList(args).contains("--with-native")) {
            System.setProperty("jitnn.withNative", "true");
            args = java.util.Arrays.stream(args).filter(a -> !a.equals("--with-native")).toArray(String[]::new);
        }
        String engine = args[0];
        Path file = Path.of(args[1]);
        boolean resnet = args[2].equals("resnet50");
        int b = Integer.parseInt(args[3]);
        int t = Integer.parseInt(args[4]);
        boolean fp16 = args[5].equals("fp16");
        String label = args[6];
        try (Model model = Model.newInstance(label, Device.gpu(), engine)) {
            // an options map, even empty: DJL's ONNX Runtime engine adds the CUDA provider only when one is given
            model.load(file.getParent(), file.getFileName().toString(), new java.util.HashMap<>());
            System.out.printf("[djl] %s: engine %s %s, %d GPUs, model on %s%s%n", label, engine, ai.djl.engine.Engine.getEngine(engine).getVersion(),
                    ai.djl.engine.Engine.getEngine(engine).getGpuCount(), model.getNDManager().getDevice(), ortRuntime());
            try (Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
                float[] pixels = resnet ? images(b) : null;
                long[] ids = resnet ? null : tokens(b, t);
                Supplier<float[]> call = () -> {
                    try (NDManager m = model.getNDManager().newSubManager()) {
                        NDArray x;
                        if (resnet && fp16) {
                            x = m.create(halves(pixels), new Shape(b, 3, 224, 224), DataType.FLOAT16);
                        } else {
                            x = resnet ? m.create(pixels, new Shape(b, 3, 224, 224)) : m.create(ids, new Shape(b, t));
                        }
                        NDArray y = predictor.predict(new NDList(x)).singletonOrThrow();
                        return y.getDataType() == DataType.FLOAT16 ? floats(y.toByteBuffer()) : y.toFloatArray();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
                time("djl-" + engine.toLowerCase(), label, b, call);
            }
        }
    }

    /** The ONNX Runtime actually loaded (version and providers), when it is on the classpath. */
    static String ortRuntime() {
        try {
            Class<?> env = Class.forName("ai.onnxruntime.OrtEnvironment");
            Object e = env.getMethod("getEnvironment").invoke(null);
            return ", ONNX Runtime " + env.getMethod("getVersion").invoke(e) + " providers " + env.getMethod("getAvailableProviders").invoke(null)
                    + " from " + env.getProtectionDomain().getCodeSource().getLocation();
        } catch (ReflectiveOperationException | RuntimeException e) {
            return "";
        }
    }

    /** FP16 bits of {@code v}, little-endian, in a direct buffer. */
    static java.nio.ByteBuffer halves(float[] v) {
        java.nio.ByteBuffer buf = java.nio.ByteBuffer.allocateDirect(v.length * 2).order(java.nio.ByteOrder.nativeOrder());
        for (float f : v) {
            buf.putShort(Float.floatToFloat16(f));
        }
        return buf.rewind();
    }

    static float[] floats(java.nio.ByteBuffer half) {
        half.order(java.nio.ByteOrder.nativeOrder());
        float[] out = new float[half.remaining() / 2];
        for (int i = 0; i < out.length; i++) {
            out[i] = Float.float16ToFloat(half.getShort());
        }
        return out;
    }

    static float[] images(int b) {
        float[] x = new float[b * 3 * 224 * 224];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) ((i * 2654435761L) % 1000) / 500.0f - 1.0f;
        }
        return x;
    }

    static long[] tokens(int b, int t) {
        long[] x = new long[b * t];
        for (int i = 0; i < x.length; i++) {
            x[i] = (i * 7919L + 13) % VOCAB;
        }
        return x;
    }

    static void time(String engine, String name, int samples, Supplier<float[]> call) {
        long c0 = System.nanoTime();
        call.get();
        double first = (System.nanoTime() - c0) / 1e6;
        for (int i = 1; i < WARMUP; i++) {
            call.get();
        }
        double[] ms = new double[RUNS];
        for (int i = 0; i < RUNS; i++) {
            long t0 = System.nanoTime();
            call.get();
            ms[i] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(ms);
        double median = ms[RUNS / 2];
        System.out.printf("[t1] %-12s %-28s median %8.3f ms  p10 %8.3f  p90 %8.3f  %10.1f samples/s  (first call %.0f ms)%n", engine, name, median, ms[RUNS / 10], ms[RUNS * 9 / 10],
                samples * 1000 / median, first);
    }
}
