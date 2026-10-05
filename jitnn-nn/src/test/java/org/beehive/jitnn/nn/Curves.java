package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/** A PyTorch reference run (metrics, then losses) and the loss-curve checks shared by the wave 4 gates. */
record Curves(float[] metrics, float[] losses) {

    static Curves load(Path file, int metrics, String script) throws Exception {
        assumeTrue(Files.exists(file), "run " + script + " to write " + file);
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        float[] m = new float[metrics];
        for (int i = 0; i < metrics; i++) {
            m[i] = buf.getFloat();
        }
        float[] l = new float[buf.remaining() / 4];
        buf.asFloatBuffer().get(l);
        return new Curves(m, l);
    }

    /** Losses {@code series}, {@code series + stride}, ... of an interleaved list (stride 2 for D and G losses). */
    static double[] series(List<Float> all, int series, int stride) {
        double[] s = new double[all.size() / stride];
        for (int i = 0; i < s.length; i++) {
            s[i] = all.get(i * stride + series);
        }
        return s;
    }

    double[] series(int series, int stride) {
        double[] s = new double[losses.length / stride];
        for (int i = 0; i < s.length; i++) {
            s[i] = losses[i * stride + series];
        }
        return s;
    }

    static double tailMean(double[] v, int tail) {
        double m = 0;
        for (int i = v.length - tail; i < v.length; i++) {
            m += v[i] / tail;
        }
        return m;
    }

    /** Step-0 loss within {@code first} relative, mean of the last {@code tail} within {@code end} relative. */
    static void check(String name, double[] mine, double[] torch, int tail, double first, double end) {
        assertTrue(mine.length == torch.length, name + ": PyTorch ran " + torch.length + " steps, jitNN " + mine.length);
        double gap = 0;
        for (int i = 0; i < Math.min(10, mine.length); i++) {
            gap = Math.max(gap, Math.abs(mine[i] - torch[i]) / Math.abs(torch[i]));
        }
        double meanJ = tailMean(mine, tail);
        double meanT = tailMean(torch, tail);
        System.out.printf("[jitnn] %s vs PyTorch: steps 0-9 gap %.2e, mean loss of the last %d steps %.4f vs %.4f%n", name, gap, tail, meanJ, meanT);
        assertTrue(Math.abs(mine[0] - torch[0]) <= first * Math.abs(torch[0]), name + ": step-0 loss " + mine[0] + " vs " + torch[0]);
        assertTrue(Math.abs(meanJ - meanT) <= end * Math.abs(meanT), name + ": final mean loss " + meanJ + " vs " + meanT);
    }
}
