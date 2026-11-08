package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import ai.beestream.jitnn.dist.NcclCollective;
import ai.beestream.jitnn.dist.ProcessGroup;
import ai.beestream.jitnn.engine.Engine;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

/**
 * Data-parallel training across two processes over NCCL: this JVM runs rank 0 on the first device
 * of {@code -Djitnn.dist.devices}, a child JVM ({@link DataParallelWorker}) rank 1 on the second,
 * and both must reproduce single-device training on the full batch. Needs
 * {@code -Djitnn.dist.nccl=true} and two CUDA devices.
 */
class DataParallelMultiProcessAccelTest {

    @AfterAll
    static void release() {
        Engine.reset();
    }

    @Test
    @SuppressWarnings("unchecked")
    void twoProcessesMatchFullBatch() throws Exception {
        assumeTrue(Boolean.getBoolean("jitnn.dist.nccl"), "no -Djitnn.dist.nccl=true");
        String[] devices = DataParallelAccelTest.devices();
        assumeTrue(devices.length == 2 && !devices[0].equals(devices[1]), "needs two distinct devices in -Djitnn.dist.devices");
        List<float[]> reference = (List<float[]>) Engine.onDevice(devices[0], () -> DataParallelAccelTest.train(0, null, 1))[1];
        Engine.reset();

        String id = NcclCollective.uniqueId();
        Path out = Files.createTempFile("jitnn-rank1", ".bin");
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.addAll(java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments());
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(DataParallelWorker.class.getName());
        cmd.addAll(List.of(id, "2", "1", devices[1], out.toString()));
        Process child = new ProcessBuilder(cmd).inheritIO().start();
        List<Object[]> mine;
        try (NcclCollective nccl = new NcclCollective(id, 2, 0, devices[0])) {
            mine = ProcessGroup.run(new String[] { devices[0] }, nccl, (r, c) -> DataParallelAccelTest.train(r, c, 2));
            Engine.reset();
        } finally {
            if (!child.waitFor(300, TimeUnit.SECONDS)) {
                child.destroyForcibly();
            }
        }
        assertEquals(0, child.exitValue(), "rank 1's process failed");
        List<float[]> rank1 = new ArrayList<>();
        try (DataInputStream in = new DataInputStream(new FileInputStream(out.toFile()))) {
            int n = in.readInt();
            for (int i = 0; i < n; i++) {
                float[] p = new float[in.readInt()];
                for (int k = 0; k < p.length; k++) {
                    p[k] = in.readFloat();
                }
                rank1.add(p);
            }
        }
        Files.deleteIfExists(out);
        List<List<float[]>> ranks = List.of((List<float[]>) mine.get(0)[1], rank1);
        for (int r = 0; r < 2; r++) {
            for (int i = 0; i < reference.size(); i++) {
                float worst = 0;
                for (int k = 0; k < reference.get(i).length; k++) {
                    worst = Math.max(worst, Math.abs(reference.get(i)[k] - ranks.get(r).get(i)[k]));
                }
                assertTrue(worst <= 2e-5f, "rank " + r + " parameter " + i + " differs from full-batch training by " + worst);
            }
        }
        System.out.printf("[dist] 2 processes on %s and %s (NCCL): match full-batch training%n", devices[0], devices[1]);
    }
}
