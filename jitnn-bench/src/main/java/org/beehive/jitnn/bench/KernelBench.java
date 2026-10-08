package org.beehive.jitnn.bench;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.beehive.jitnn.engine.Engine;
import org.beehive.jitnn.engine.Node;

/**
 * Times every distinct operator call of the reference workloads on the current device, following
 * the kernel optimisation plan: correctness is the goldens' job, so here each call is replayed on
 * made-up inputs; 20 warm-up calls, then at least 100 timed calls and 0.3 s; the median, p10 and
 * p90 of the per-call device time (TornadoVM's profiler) and of the host-to-host time; a window is
 * accepted when (p90 - p10) / median is under 5% (or p90 - p10 is at most 2 us, the profiler's
 * resolution), no clock sample in it shows a power or thermal
 * cap and the mean SM clock is at least 1,350 MHz, else retried after 10 s, up to three times.
 * One JSON line per call goes to the output file.
 *
 * <p>Device times need {@code -Djitnn.profile=true} (and on CUDA {@code -Djitnn.cudaGraph=false});
 * without the profiler the run measures host-to-host times only, free of the profiler's cost. Properties:
 * {@code jitnn.kbench.out} (file, appended), {@code jitnn.kbench.workloads} and
 * {@code jitnn.kbench.families} (comma-separated filters), {@code jitnn.kbench.from} (skip the first
 * n calls, to resume a run), {@code jitnn.kbench.label} (free text
 * recorded with every line, e.g. the commits).
 */
public final class KernelBench {

    static final int WARMUP = Integer.getInteger("jitnn.kbench.warmup", 20);
    static final int MIN_CALLS = Integer.getInteger("jitnn.kbench.calls", 100);
    static final long MIN_WINDOW_NS = 300_000_000L;
    static final int MAX_CALLS = 2000;
    static final double MAX_SPREAD = 0.05;
    /** The profiler's timestamps step by about 1 us: a spread of up to 2 us is resolution, not noise. */
    static final double MIN_SPREAD_US = 2.0;
    static final int MIN_MHZ = Integer.getInteger("jitnn.kbench.minMhz", 1350);
    static final boolean PROFILE = Boolean.getBoolean("jitnn.profile");
    static final double BANDWIDTH = Double.parseDouble(System.getProperty("jitnn.kbench.bandwidth", "1721e9"));
    static final double FP32_PEAK = Double.parseDouble(System.getProperty("jitnn.kbench.fp32", "19.0e12"));
    static final double HALF_PEAK = Double.parseDouble(System.getProperty("jitnn.kbench.half", "245e12"));

    private KernelBench() {
    }

    /** The distinct calls of the workloads, in first-seen order, with per-workload use counts. */
    static List<OpCase> cases(Map<String, List<Node>> captured, String families) {
        Map<String, OpCase> byKey = new LinkedHashMap<>();
        List<String> fam = families.isEmpty() ? List.of() : Arrays.asList(families.split(","));
        for (var w : captured.entrySet()) {
            for (Node n : w.getValue()) {
                OpCase c = new OpCase(n);
                if (!fam.isEmpty() && !fam.contains(OpCase.family(c.op.name()))) {
                    continue;
                }
                OpCase seen = byKey.computeIfAbsent(c.key, k -> c);
                seen.uses.merge(w.getKey(), 1, Integer::sum);
            }
        }
        return new ArrayList<>(byKey.values());
    }

    record Stats(double median, double p10, double p90) {
        static Stats of(double[] v) {
            double[] s = v.clone();
            Arrays.sort(s);
            return new Stats(s[s.length / 2], s[s.length / 10], s[s.length * 9 / 10]);
        }

        double spread() {
            return median > 0 ? (p90 - p10) / median : 0;
        }
    }

    record Result(Stats device, Stats host, double launches, ClockMonitor.Window window, int attempts, int calls, boolean accepted) {
    }

    static Result time(OpCase c, ClockMonitor clocks) {
        Result last = null;
        for (int attempt = 1; attempt <= 3; attempt++) {
            for (int i = 0; i < WARMUP; i++) {
                c.record();
                Engine.sync();
            }
            Engine.kernelBreakdown();
            List<Double> dev = new ArrayList<>();
            List<Double> host = new ArrayList<>();
            double launches = 0;
            long start = System.nanoTime();
            while ((dev.size() < MIN_CALLS || System.nanoTime() - start < MIN_WINDOW_NS) && dev.size() < MAX_CALLS) {
                long t0 = System.nanoTime();
                c.record();
                Engine.sync();
                host.add((System.nanoTime() - t0) / 1e3);
                long ns = 0;
                for (long[] v : Engine.kernelBreakdown().values()) {
                    launches += v[0];
                    ns += v[1];
                }
                dev.add(ns / 1e3);
            }
            long end = System.nanoTime();
            double[] d = dev.stream().mapToDouble(Double::doubleValue).toArray();
            double[] h = host.stream().mapToDouble(Double::doubleValue).toArray();
            Stats sd = Stats.of(d);
            ClockMonitor.Window w = clocks.available() ? clocks.window(start, end) : new ClockMonitor.Window(0, 0, false);
            boolean clockOk = !clocks.available() || (!w.throttled() && w.meanMhz() >= MIN_MHZ);
            Stats sh = Stats.of(h);
            // without the profiler (host-only run) the host time is what is judged
            Stats judged = PROFILE ? sd : sh;
            boolean ok = clockOk && (judged.spread() < MAX_SPREAD || judged.p90() - judged.p10() <= MIN_SPREAD_US);
            last = new Result(sd, sh, launches / d.length, w, attempt, d.length, ok);
            if (ok) {
                return last;
            }
            try {
                // a capped clock needs the GPU to cool and recover; a noisy window just another try
                Thread.sleep(clockOk ? 2_000 : 10_000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return last;
            }
        }
        return last;
    }

    static String json(OpCase c, Result r, String device, String label) {
        StringBuilder b = new StringBuilder(512);
        double devS = r.device().median() * 1e-6;
        double flops = c.flops();
        double gbs = devS > 0 ? c.bytes() / devS : 0;
        double tf = devS > 0 ? flops / devS : 0;
        b.append('{');
        kv(b, "label", label).append(',');
        kv(b, "device", device).append(',');
        kv(b, "op", c.op.name()).append(',');
        kv(b, "family", OpCase.family(c.op.name())).append(',');
        b.append("\"iattr\":").append(Arrays.toString(c.iattr)).append(',');
        b.append("\"fattr\":").append(floats(c.fattr)).append(',');
        b.append("\"in\":").append(shapes(c.inShapes, c.inTypes)).append(',');
        b.append("\"out\":").append(shapes(c.outShapes, c.outTypes)).append(',');
        b.append("\"uses\":{");
        int k = 0;
        for (var u : c.uses.entrySet()) {
            b.append(k++ > 0 ? "," : "").append('"').append(u.getKey()).append("\":").append(u.getValue());
        }
        b.append("},");
        num(b, "device_us", r.device().median()).append(',');
        num(b, "device_us_p10", r.device().p10()).append(',');
        num(b, "device_us_p90", r.device().p90()).append(',');
        num(b, "host_us", r.host().median()).append(',');
        num(b, "host_us_p10", r.host().p10()).append(',');
        num(b, "host_us_p90", r.host().p90()).append(',');
        num(b, "launches", r.launches()).append(',');
        b.append("\"bytes\":").append(c.bytes()).append(',');
        num(b, "flops", flops).append(',');
        num(b, "gb_per_s", gbs / 1e9).append(',');
        num(b, "bandwidth_eff", gbs / BANDWIDTH).append(',');
        num(b, "tflops", tf / 1e12).append(',');
        num(b, "compute_eff", flops > 0 ? tf / (c.half() ? HALF_PEAK : FP32_PEAK) : 0).append(',');
        b.append("\"half\":").append(c.half()).append(',');
        num(b, "clock_mhz", r.window().meanMhz()).append(',');
        b.append("\"clock_samples\":").append(r.window().samples()).append(',');
        b.append("\"power_capped\":").append(r.window().throttled()).append(',');
        num(b, "spread", r.device().spread()).append(',');
        b.append("\"calls\":").append(r.calls()).append(',');
        b.append("\"attempts\":").append(r.attempts()).append(',');
        b.append("\"accepted\":").append(r.accepted());
        return b.append('}').toString();
    }

    private static StringBuilder kv(StringBuilder b, String k, String v) {
        return b.append('"').append(k).append("\":\"").append(v.replace("\\", "\\\\").replace("\"", "\\\"")).append('"');
    }

    private static StringBuilder num(StringBuilder b, String k, double v) {
        return b.append('"').append(k).append("\":").append(Double.isFinite(v) ? String.format(Locale.ROOT, "%.6g", v) : "null");
    }

    private static String floats(float[] f) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < f.length; i++) {
            b.append(i > 0 ? "," : "").append(Float.isFinite(f[i]) ? Float.toString(f[i]) : "null");
        }
        return b.append(']').toString();
    }

    private static String shapes(int[][] dims, org.beehive.jitnn.DType[] types) {
        StringBuilder b = new StringBuilder("[");
        for (int i = 0; i < dims.length; i++) {
            b.append(i > 0 ? "," : "").append("{\"shape\":").append(Arrays.toString(dims[i])).append(",\"dtype\":\"").append(types[i]).append("\"}");
        }
        return b.append(']').toString();
    }

    /** Captures the workloads, times every distinct call and appends the results; returns how many were written. */
    public static int run() throws IOException {
        // without -Djitnn.profile=true only host-to-host times are measured (device fields are 0):
        // the profiler's own cost would otherwise be part of the host time

        Path out = Path.of(System.getProperty("jitnn.kbench.out", "kernels.jsonl"));
        String label = System.getProperty("jitnn.kbench.label", "");
        String device = System.getProperty("jitnn.device", "default");
        Map<String, List<Node>> captured = Workloads.capture(System.getProperty("jitnn.kbench.workloads", ""));
        List<OpCase> cases = cases(captured, System.getProperty("jitnn.kbench.families", ""));
        System.out.printf("[kbench] %d workloads, %d distinct calls%n", captured.size(), cases.size());
        int written = 0;
        try (ClockMonitor clocks = new ClockMonitor(System.getProperty("jitnn.kbench.gpu", "0"));
                PrintWriter w = new PrintWriter(Files.newBufferedWriter(out, StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
            int i = 0;
            int from = Integer.getInteger("jitnn.kbench.from", 0);
            for (OpCase c : cases) {
                i++;
                if (i <= from || !c.replayable()) {
                    continue;
                }
                try {
                    Result r = time(c, clocks);
                    w.println(json(c, r, device, label));
                    w.flush();
                    written++;
                    System.out.printf(Locale.ROOT, "[kbench] %4d/%d %-22s %10.2f us  %s%n", i, cases.size(), c.op.name(), r.device().median(), r.accepted() ? "" : "NOT ACCEPTED");
                } catch (RuntimeException e) {
                    System.out.printf("[kbench] %4d/%d %-22s failed: %s%n", i, cases.size(), c.op.name(), e.getMessage());
                } finally {
                    c.release();
                    Engine.reset();
                }
            }
        }
        return written;
    }
}
