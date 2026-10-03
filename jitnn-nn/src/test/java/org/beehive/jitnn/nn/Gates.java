package org.beehive.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.beehive.jitnn.Tensor;

/** Checks shared by the model gates. */
final class Gates {

    private Gates() {
    }

    /**
     * Every parameter's step-0 gradient against PyTorch's FP64 gradient (scripts/pytorch/cnn_grad64.py, transformers.py grad64).
     * FP32 is only good to about 1e-3 relative through 20 BatchNorm layers, so a parameter passes when
     * its error is within three times PyTorch's own FP32 error on it, either relative to its gradient
     * or in absolute terms against the model's median absolute FP32 error. The absolute floor keeps
     * gradients that nearly cancel to zero (a BatchNorm scale whose terms sum to ~1e-3 of its
     * neighbours') from failing on noise far below everyone else's.
     */
    static void checkGradients(String name, List<Tensor> params, String prefix) throws Exception {
        Path f64 = Path.of(prefix + ".grad0_f64");
        Path f32err = Path.of(prefix + ".grad0_f32err");
        assumeTrue(Files.exists(f64) && Files.exists(f32err), "run the FP64 gradient script for " + prefix);
        java.nio.DoubleBuffer truth = ByteBuffer.wrap(Files.readAllBytes(f64)).order(ByteOrder.LITTLE_ENDIAN).asDoubleBuffer();
        java.nio.FloatBuffer torchErr = ByteBuffer.wrap(Files.readAllBytes(f32err)).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        int count = params.size();
        double[] rel = new double[count];
        double[] abs = new double[count];
        double[] torchAbs = new double[count];
        for (int i = 0; i < count; i++) {
            float[] g = params.get(i).grad().toFloatArray();
            double diff = 0;
            double norm = 0;
            for (float v : g) {
                double t = truth.get();
                diff += (v - t) * (v - t);
                norm += t * t;
            }
            abs[i] = Math.sqrt(diff);
            rel[i] = abs[i] / Math.max(Math.sqrt(norm), 1e-30);
            torchAbs[i] = torchErr.get(i) * Math.sqrt(norm);
        }
        assertTrue(!truth.hasRemaining(), name + ": PyTorch has more gradient values than jitNN parameters");
        double[] sorted = torchAbs.clone();
        java.util.Arrays.sort(sorted);
        double floor = 3 * sorted[count / 2];
        double worstRatio = 0;
        int worstAt = 0;
        for (int i = 0; i < count; i++) {
            double ratio = Math.min(rel[i] / (3 * torchErr.get(i) + 1e-5), abs[i] / Math.max(floor, 1e-30));
            if (ratio > worstRatio) {
                worstRatio = ratio;
                worstAt = i;
            }
        }
        System.out.printf("[jitnn] %s step-0 gradients of %d parameters vs FP64: worst parameter %d at %.2f of its bound (relative %.2e, PyTorch FP32 %.2e; absolute %.2e, floor %.2e)%n",
                name, count, worstAt, worstRatio, rel[worstAt], torchErr.get(worstAt), abs[worstAt], floor);
        assertTrue(worstRatio <= 1, name + ": step-0 gradient of parameter " + worstAt + " is off by " + rel[worstAt] + " relative, " + abs[worstAt] + " absolute");
    }
}
