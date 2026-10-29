package org.beehive.jitnn;

import static org.beehive.jitnn.Checks.close;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.function.Supplier;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Engine.Backend;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * The CPU backend with an OpenCL device on the host CPU (PoCL, the Intel CPU runtime): operators
 * without a CPU kernel run there instead of their reference. Skipped when there is no such device
 * (or the JVM runs without {@code LD_PRELOAD=$JAVA_HOME/lib/libjsig.so}). The operators go to the
 * device without timing ({@code jitnn.cpu.opencl.choose=always}) unless a test says otherwise.
 */
class CpuOpenClAccelTest {

    @BeforeAll
    static void device() {
        Assumptions.assumeTrue(Engine.cpuOpenClDevice() != null, "no OpenCL device on the host CPU");
        System.setProperty("jitnn.cpu.opencl.choose", "always");
    }

    @AfterAll
    static void reset() {
        System.clearProperty("jitnn.cpu.opencl.choose");
    }

    @TestFactory
    List<DynamicTest> cpuWithOpenClMatchesPyTorch() {
        return GoldenOps.tests(Backend.CPU);
    }

    @Test
    void tensorsMoveBetweenKernelsAndTheDevice() {
        // linear (CPU kernel), then upsampling and a permutation (no CPU kernel), then CPU kernels again
        Supplier<float[]> step = () -> {
            Tensor x = Tensor.uniform(1, -1, 1, 2, 3, 8, 8);
            Tensor w = Tensor.uniform(2, -0.3f, 0.3f, 8, 8);
            Tensor u = F.interpolate(F.linear(x, w, null), 16, 16, true, false).permute(0, 2, 3, 1);
            Tensor y = u.relu().add(u);
            float[] a = y.toFloatArray();
            float[] all = new float[a.length + 1];
            System.arraycopy(a, 0, all, 0, a.length);
            all[a.length] = y.sum().item();
            return all;
        };
        float[] expected = Engine.with(Backend.REFERENCE, step::get);
        long before = openClCalls();
        for (int i = 0; i < 3; i++) {
            // the same trace again: buffers of the previous step are reused on both sides
            close(expected, Engine.with(Backend.CPU, step::get), 1e-5, 1e-5, "step " + i);
        }
        assertTrue(openClCalls() > before, "upsampling and permute ran on the OpenCL device");
    }

    @Test
    void operatorsTheDeviceCannotCompileRunTheirReference() {
        // FP16 needs cl_khr_fp16, which some CPU drivers lack: the result is the same either way
        Supplier<float[]> step = () -> Tensor.uniform(3, -4, 4, 1000).half().to(DType.F32).mul(Tensor.uniform(4, -1, 1, 1000)).toFloatArray();
        float[] expected = Engine.with(Backend.REFERENCE, step::get);
        for (int i = 0; i < 2; i++) {
            close(expected, Engine.with(Backend.CPU, step::get), 0, 0, "step " + i);
        }
    }

    @Test
    void timedChoiceGivesTheSameResults() {
        // the default: the reference twice, the device three times, then the faster of the two
        Supplier<float[]> step = () -> F.interpolate(Tensor.uniform(5, -1, 1, 4, 8, 32, 32).relu(), 64, 64, false, false).permute(0, 3, 1, 2).sum(1, false).toFloatArray();
        float[] expected = Engine.with(Backend.REFERENCE, step::get);
        System.clearProperty("jitnn.cpu.opencl.choose");
        try {
            for (int i = 0; i < 7; i++) {
                close(expected, Engine.with(Backend.CPU, step::get), 1e-5, 1e-5, "step " + i);
            }
        } finally {
            System.setProperty("jitnn.cpu.opencl.choose", "always");
        }
    }

    private static long openClCalls() {
        long n = 0;
        for (long[] v : Engine.cpuStats().values()) {
            n += v[3];
        }
        return n;
    }
}
