package ai.beestream.jitnn.bench;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

import ai.beestream.jitnn.engine.Engine;

/**
 * Runs each workload's step end to end, as the models really run (fusion, plan replay, buffer
 * reuse), and records per step: with {@code -Djitnn.profile=true}, the device time of every
 * operator (TornadoVM's profiler); without it, the step's wall time. 10 warm-up steps, then the
 * median over 30. One JSON line per workload to {@code jitnn.kbench.out}.
 */
public final class ModelProfile {

    static final int WARMUP = Integer.getInteger("jitnn.kbench.modelWarmup", 10);
    static final int STEPS = Integer.getInteger("jitnn.kbench.steps", 30);

    private ModelProfile() {
    }

    public static int run() throws IOException {
        boolean profile = Boolean.getBoolean("jitnn.profile");
        Path out = Path.of(System.getProperty("jitnn.kbench.out", "models.jsonl"));
        String filter = System.getProperty("jitnn.kbench.workloads", "");
        String label = System.getProperty("jitnn.kbench.label", "");
        int written = 0;
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(out, StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            for (Workloads.Workload wl : Workloads.all()) {
                if (!filter.isEmpty() && !Arrays.asList(filter.split(",")).contains(wl.name())) {
                    continue;
                }
                Runnable step = wl.setup().get();
                for (int i = 0; i < WARMUP; i++) {
                    step.run();
                    Engine.sync();
                }
                Engine.kernelBreakdown();
                boolean cpuStats = Boolean.getBoolean("jitnn.cpu.stats");
                if (cpuStats) {
                    Engine.cpuReport();
                }
                double[] wall = new double[STEPS];
                Map<String, double[]> perOp = new TreeMap<>();
                for (int i = 0; i < STEPS; i++) {
                    long t0 = System.nanoTime();
                    step.run();
                    Engine.sync();
                    wall[i] = (System.nanoTime() - t0) / 1e6;
                    for (var e : Engine.kernelBreakdown().entrySet()) {
                        String op = e.getKey().replaceAll("[\\[#].*", "");
                        double[] acc = perOp.computeIfAbsent(op, k -> new double[2]);
                        acc[0] += e.getValue()[0];
                        acc[1] += e.getValue()[1] / 1e6;
                    }
                }
                Arrays.sort(wall);
                System.out.printf(Locale.ROOT, "[model] %-28s median %9.3f ms/step  p10 %9.3f  p90 %9.3f%n", wl.name(), wall[STEPS / 2], wall[STEPS / 10], wall[STEPS * 9 / 10]);
                if (cpuStats) {
                    System.out.print(Engine.cpuReport());
                }
                StringBuilder b = new StringBuilder();
                b.append(String.format(Locale.ROOT, "{\"label\":\"%s\",\"device\":\"%s\",\"workload\":\"%s\",\"profile\":%b,\"step_ms\":%.4f,\"step_ms_p10\":%.4f,\"step_ms_p90\":%.4f,\"ops\":{", label,
                        System.getProperty("jitnn.device", "default"), wl.name(), profile, wall[STEPS / 2], wall[STEPS / 10], wall[STEPS * 9 / 10]));
                int k = 0;
                double dev = 0;
                for (var e : perOp.entrySet()) {
                    double ms = e.getValue()[1] / STEPS;
                    dev += ms;
                    b.append(k++ > 0 ? "," : "").append(String.format(Locale.ROOT, "\"%s\":{\"family\":\"%s\",\"launches\":%.1f,\"device_ms\":%.4f}", e.getKey(), OpCase.family(e.getKey()),
                            e.getValue()[0] / STEPS, ms));
                }
                b.append(String.format(Locale.ROOT, "},\"device_ms\":%.4f}", dev));
                w.println(b);
                w.flush();
                written++;
                System.out.printf(Locale.ROOT, "[models] %-24s step %8.2f ms  device %8.2f ms%n", wl.name(), wall[STEPS / 2], dev);
                Engine.reset();
            }
        }
        return written;
    }
}
