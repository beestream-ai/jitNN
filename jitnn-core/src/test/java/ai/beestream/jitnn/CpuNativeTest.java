package ai.beestream.jitnn;

import static ai.beestream.jitnn.Checks.close;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import ai.beestream.jitnn.cpu.Blas;
import ai.beestream.jitnn.cpu.Bnns;
import ai.beestream.jitnn.cpu.Dnnl;
import ai.beestream.jitnn.cpu.Race;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * The CPU backend with native libraries: a BLAS ({@code jitnn.cpu.blas}) for linear, matmul and
 * convolutions, oneDNN ({@code jitnn.cpu.onednn}) and BNNS ({@code jitnn.cpu.bnns}, macOS) for
 * convolutions and their gradients. Each gives
 * what the backend's own kernels give. Tests of a library that is not found are skipped.
 */
class CpuNativeTest {

    @AfterEach
    void reset() {
        System.clearProperty("jitnn.cpu.native.choose");
        Race.reset();
    }

    @TestFactory
    List<DynamicTest> cpuWithBlasMatchesPyTorch() {
        Assumptions.assumeTrue(Blas.library() != null, "no BLAS library");
        System.setProperty("jitnn.cpu.native.choose", "blas");
        return GoldenOps.tests(Backend.CPU);
    }

    @TestFactory
    List<DynamicTest> cpuWithOneDnnMatchesPyTorch() {
        Assumptions.assumeTrue(Dnnl.available(), "no oneDNN library");
        System.setProperty("jitnn.cpu.native.choose", "onednn");
        return GoldenOps.tests(Backend.CPU);
    }

    @TestFactory
    List<DynamicTest> cpuWithBnnsMatchesPyTorch() {
        Assumptions.assumeTrue(Bnns.available(), "no BNNS");
        System.setProperty("jitnn.cpu.native.choose", "bnns");
        return GoldenOps.tests(Backend.CPU);
    }

    @Test
    void bnnsMatchesTheKernels() {
        Assumptions.assumeTrue(Bnns.available(), "no BNNS");
        sameAsKernels("bnns");
    }

    @Test
    void blasMatchesTheKernels() {
        Assumptions.assumeTrue(Blas.library() != null, "no BLAS library");
        sameAsKernels("blas");
    }

    @Test
    void oneDnnMatchesTheKernels() {
        Assumptions.assumeTrue(Dnnl.available(), "no oneDNN library");
        sameAsKernels("onednn");
    }

    /** Convolutions (padded, strided, dilated, grouped, with bias and ReLU), their gradients, linear and matmul. */
    private static void sameAsKernels(String library) {
        Supplier<float[]> step = () -> {
            Tensor x = Tensor.uniform(1, -1, 1, 3, 8, 13, 11).requiresGrad(true);
            Tensor w3 = Tensor.parameter(Tensor.uniform(2, -0.3f, 0.3f, 6, 8, 3, 3));
            Tensor b = Tensor.parameter(Tensor.uniform(3, -0.3f, 0.3f, 6));
            Tensor w1 = Tensor.parameter(Tensor.uniform(4, -0.3f, 0.3f, 4, 6, 1, 1));
            Tensor wg = Tensor.parameter(Tensor.uniform(5, -0.3f, 0.3f, 8, 2, 3, 3));
            Tensor c = F.conv2dRelu(x, w3, b, 1, 1, 1, 1);
            Tensor s = F.conv2d(x, w3, null, 2, 2, 2, 1);
            Tensor g = F.conv2d(x, wg, null, 1, 1, 1, 4);
            Tensor p = F.conv2d(c, w1, null, 1, 0, 1, 1);
            Tensor l = F.linear(p.reshape(3 * 4 * 13, 11), Tensor.uniform(6, -0.3f, 0.3f, 9, 11), Tensor.uniform(7, -0.3f, 0.3f, 9));
            Tensor m = l.matmul(l, true, false);
            Tensor loss = m.sum().add(s.mul(s).sum()).add(g.mul(g).sum());
            loss.backward();
            List<float[]> parts = new ArrayList<>(List.of(c.toFloatArray(), s.toFloatArray(), g.toFloatArray(), p.toFloatArray(), l.toFloatArray(), m.toFloatArray()));
            for (Tensor t : new Tensor[] { x, w3, b, w1, wg }) {
                parts.add(t.grad().toFloatArray());
            }
            return concat(parts);
        };
        System.setProperty("jitnn.cpu.native.choose", "kernel");
        float[] kernels = Engine.with(Backend.CPU, step::get);
        System.setProperty("jitnn.cpu.native.choose", library);
        for (int i = 0; i < 2; i++) {
            close(kernels, Engine.with(Backend.CPU, step::get), 1e-4, 1e-4, library + " step " + i);
        }
        // timed: whichever way wins, the results stay the same
        System.setProperty("jitnn.cpu.native.choose", "measure");
        for (int i = 0; i < 10; i++) {
            close(kernels, Engine.with(Backend.CPU, step::get), 1e-4, 1e-4, "timed step " + i);
        }
    }

    private static float[] concat(List<float[]> parts) {
        int size = 0;
        for (float[] p : parts) {
            size += p.length;
        }
        float[] all = new float[size];
        int at = 0;
        for (float[] p : parts) {
            System.arraycopy(p, 0, all, at, p.length);
            at += p.length;
        }
        return all;
    }
}
