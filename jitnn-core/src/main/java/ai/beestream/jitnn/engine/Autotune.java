package ai.beestream.jitnn.engine;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Grad;
import ai.beestream.jitnn.Shape;
import ai.beestream.jitnn.Tensor;

/**
 * Run-time autotuning: operators that can run a shape with several kernel variants ask
 * {@link #choose} which one, instead of applying their size rules. The first time a shape is
 * seen ({@code -Djitnn.autotune=on}) the rule's variant runs, and once the plan has run the
 * choice is measured: the operator is replayed alone on random inputs with each variant, the
 * outputs are checked against the rule's, and a variant faster by more than
 * {@code jitnn.autotune.margin} (5%) replaces the rule's. Plans that used the choice are
 * recompiled with it.
 *
 * <p>Decisions are kept per device in {@code jitnn.autotune.dir} (default
 * {@code ~/.jitnn/autotune}, one properties file per device) and read back by later runs;
 * {@code -Djitnn.autotune=use} applies saved decisions without measuring new shapes, and
 * {@code off} (the default) ignores them.
 */
public final class Autotune {

    private static volatile String MODE = System.getProperty("jitnn.autotune", "off").trim();
    private static volatile Path DIR = Path.of(System.getProperty("jitnn.autotune.dir", Path.of(System.getProperty("user.home"), ".jitnn", "autotune").toString()));
    private static final double MARGIN = Double.parseDouble(System.getProperty("jitnn.autotune.margin", "0.05"));
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.autotune.verbose");
    /** Operator copies per timed plan, so the plan's fixed cost is shared (fewer for large outputs). */
    private static final int REPS = Integer.getInteger("jitnn.autotune.reps", 8);
    /** Bytes of outputs a measurement plan may hold: large operators get fewer copies. */
    private static final long BUDGET = 256L << 20;
    private static final int RUNS = Integer.getInteger("jitnn.autotune.runs", 15);

    /** Decisions per device file: choice key to variant. */
    private static final Map<Path, Map<String, String>> DECISIONS = new ConcurrentHashMap<>();
    /** Variants forced on this thread while a choice is measured. */
    private static final ThreadLocal<Map<String, String>> FORCED = ThreadLocal.withInitial(Map::of);
    /** Choices met on this thread that wait to be measured, by key. */
    private static final ThreadLocal<Map<String, Pending>> PENDING = ThreadLocal.withInitial(LinkedHashMap::new);
    private static final ThreadLocal<Boolean> TUNING = ThreadLocal.withInitial(() -> false);

    /** A choice to measure: its candidates, and the operator to replay. */
    private record Pending(String key, String fallback, String[] candidates, Op op, int[] iattr, float[] fattr, Shape[] inShapes, DType[] inTypes, Shape[] outShapes,
            DType[] outTypes) {
    }

    private Autotune() {
    }

    /** Sets the mode, as {@code -Djitnn.autotune}: "on", "use" or "off". */
    public static void setMode(String mode) {
        if (!mode.equals("on") && !mode.equals("use") && !mode.equals("off")) {
            throw new IllegalArgumentException("autotune mode " + mode + ": on, use or off");
        }
        MODE = mode;
    }

    /** Sets the directory of the decision files, as {@code -Djitnn.autotune.dir}; decisions loaded so far are dropped. */
    public static void setDirectory(Path dir) {
        DIR = dir;
        DECISIONS.clear();
    }

    /** True when choices are tuned or saved decisions applied. */
    public static boolean enabled() {
        return !MODE.equals("off");
    }

    /**
     * The variant to emit for choice {@code family:shape} of the node being emitted: a forced one
     * while it is measured, else the saved decision, else {@code fallback} (the operator's rule),
     * recording the choice for measurement when tuning is on.
     */
    public static String choose(Emitter e, String family, String shape, String fallback, String... candidates) {
        String key = family + ":" + shape;
        String forced = FORCED.get().get(key);
        if (forced != null) {
            return forced;
        }
        if (!enabled() || candidates.length < 2) {
            return fallback;
        }
        String decided = decisions().get(key);
        if (decided != null && Arrays.asList(candidates).contains(decided)) {
            return decided;
        }
        Node n = e.node();
        // deterministic runs never measure: the choice would depend on this run's timings
        if (MODE.equals("on") && !Engine.deterministic() && !TUNING.get() && n != null && replayable(n)) {
            e.choices.add(key);
            PENDING.get().putIfAbsent(key, new Pending(key, fallback, candidates.clone(), n.op, n.iattr == null ? null : n.iattr.clone(), n.fattr == null ? null : n.fattr.clone(),
                    shapes(n.inputs), types(n.inputs), shapes(n.outputs), types(n.outputs)));
        }
        return fallback;
    }

    /** The forced variants, for the plan cache key (so measurement plans never stand in for real ones). */
    static String keyTag() {
        Map<String, String> f = FORCED.get();
        return f.isEmpty() ? "" : "AT" + new TreeMap<>(f) + "|";
    }

    private static boolean replayable(Node n) {
        for (Tensor t : n.inputs) {
            if (t == null || t.dtype() != DType.F32 && t.dtype() != DType.F16) {
                return false;
            }
        }
        for (Tensor t : n.outputs) {
            if (t.dtype() != DType.F32 && t.dtype() != DType.F16) {
                return false;
            }
        }
        return true;
    }

    private static Shape[] shapes(Tensor[] ts) {
        Shape[] s = new Shape[ts.length];
        for (int i = 0; i < ts.length; i++) {
            s[i] = ts[i].shape();
        }
        return s;
    }

    private static DType[] types(Tensor[] ts) {
        DType[] d = new DType[ts.length];
        for (int i = 0; i < ts.length; i++) {
            d[i] = ts[i].dtype();
        }
        return d;
    }

    // ---------------------------------------------------------------- measuring

    /** Measures this thread's pending choices (called by the executor once a plan has run, with no lock held). */
    static void tunePending() {
        Map<String, Pending> pending = PENDING.get();
        if (pending.isEmpty() || TUNING.get()) {
            return;
        }
        List<Pending> todo = new ArrayList<>(pending.values());
        pending.clear();
        List<Node> saved = Trace.drain();
        TUNING.set(true);
        java.util.Set<String> decided = new java.util.HashSet<>();
        try (Grad.Scope off = Grad.disabled()) {
            for (Pending p : todo) {
                if (decisions().containsKey(p.key)) {
                    continue;
                }
                decide(p);
                decided.add(p.key);
            }
        } finally {
            TUNING.set(false);
            FORCED.set(Map.of());
            TornadoExecutor.forgetTuning(decided);
            Trace.drain();
            for (Node n : saved) {
                Trace.record(n);
            }
        }
    }

    private static void decide(Pending p) {
        long outBytes = 0;
        for (int i = 0; i < p.outShapes.length; i++) {
            outBytes += (long) p.outShapes[i].numel() * p.outTypes[i].bytes();
        }
        int reps = (int) Math.max(1, Math.min(REPS, BUDGET / Math.max(outBytes, 1)));
        Tensor[] inputs = new Tensor[p.inShapes.length];
        java.util.Random rnd = new java.util.Random(p.key.hashCode());
        for (int i = 0; i < inputs.length; i++) {
            float[] v = new float[p.inShapes[i].numel()];
            for (int j = 0; j < v.length; j++) {
                v[j] = rnd.nextFloat() * 2 - 1;
            }
            int[] dims = p.inShapes[i].dims();
            inputs[i] = p.inTypes[i] == DType.F16 ? Tensor.persistentHalf(v, dims) : Tensor.persistent(Tensor.of(v, dims));
        }
        float[] reference = null;
        Map<String, Double> times = new LinkedHashMap<>();
        // the rule's variant first: its output is the reference the others are checked against
        List<String> order = new ArrayList<>();
        order.add(p.fallback);
        for (String c : p.candidates) {
            if (!c.equals(p.fallback)) {
                order.add(c);
            }
        }
        for (String variant : order) {
            FORCED.set(Map.of(p.key, variant));
            try {
                Tensor[] first = replay(p, inputs);
                Engine.sync(first);
                float[] out = first[0].toFloatArray();
                if (reference == null) {
                    reference = out;
                } else if (!close(reference, out)) {
                    if (VERBOSE) {
                        System.err.printf("[jitnn] autotune %s: %s gives different results, skipped%n", p.key, variant);
                    }
                    continue;
                }
                times.put(variant, time(p, inputs, reps));
            } catch (RuntimeException ex) {
                if (VERBOSE) {
                    System.err.printf("[jitnn] autotune %s: %s failed (%s), skipped%n", p.key, variant, ex);
                }
            }
        }
        FORCED.set(Map.of());
        // the measurement plans and inputs go now: the next choice may need the device memory
        TornadoExecutor.forgetTuning(java.util.Set.of());
        for (Tensor t : inputs) {
            t.release();
        }
        Double base = times.get(p.fallback);
        String best = p.fallback;
        if (base != null) {
            for (Map.Entry<String, Double> t : times.entrySet()) {
                if (t.getValue() < times.get(best) && t.getValue() < base * (1 - MARGIN)) {
                    best = t.getKey();
                }
            }
        }
        if (VERBOSE) {
            StringBuilder b = new StringBuilder();
            times.forEach((v, t) -> b.append(String.format(" %s %.1f us", v, t * 1e6)));
            System.err.printf("[jitnn] autotune %s:%s -> %s%n", p.key, b, best);
        }
        record(p.key, best);
    }

    /** Records the operator once on the given inputs; returns its outputs. */
    private static Tensor[] replay(Pending p, Tensor[] inputs) {
        Tensor[] outs = new Tensor[p.outShapes.length];
        for (int i = 0; i < outs.length; i++) {
            outs[i] = Tensor.result(p.outShapes[i], p.outTypes[i]);
        }
        Tensor.record(p.op, inputs, outs, p.iattr, p.fattr, true);
        return outs;
    }

    /**
     * Seconds per operator: a plan of {@code reps} copies is compiled and run once, then executed again
     * RUNS times as it is (median), so recording and plan lookup are not counted.
     */
    private static double time(Pending p, Tensor[] inputs, int reps) {
        List<Tensor> outs = new ArrayList<>();
        for (int i = 0; i < reps; i++) {
            outs.addAll(Arrays.asList(replay(p, inputs)));
        }
        Engine.sync(outs.toArray(new Tensor[0]));
        return TornadoExecutor.timeLast(RUNS) / reps;
    }

    private static boolean close(float[] a, float[] b) {
        if (a.length != b.length) {
            return false;
        }
        float scale = 1e-6f;
        for (float v : a) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < a.length; i++) {
            if (!(Math.abs(a[i] - b[i]) <= 1e-3f * scale)) {
                return false;
            }
        }
        return true;
    }

    // ---------------------------------------------------------------- decisions on disk

    /** The decisions of the current device. */
    public static Map<String, String> decisions() {
        return DECISIONS.computeIfAbsent(file(), Autotune::load);
    }

    /** The current device's decisions file. */
    public static Path file() {
        return DIR.resolve(Tuning.fileName().replace(".json", ".properties"));
    }

    private static Map<String, String> load(Path file) {
        Map<String, String> m = new ConcurrentHashMap<>();
        if (Files.exists(file)) {
            Properties p = new Properties();
            try (InputStream in = Files.newInputStream(file)) {
                p.load(in);
            } catch (IOException e) {
                System.err.println("[jitnn] autotune: cannot read " + file + ": " + e);
            }
            p.forEach((k, v) -> m.put(k.toString(), v.toString()));
        }
        return m;
    }

    private static synchronized void record(String key, String variant) {
        Map<String, String> d = decisions();
        d.put(key, variant);
        Path f = file();
        try {
            Files.createDirectories(f.getParent());
            Properties p = new Properties();
            // merge what other processes saved meanwhile
            load(f).forEach(p::setProperty);
            d.forEach(p::setProperty);
            Path tmp = Files.createTempFile(f.getParent(), "autotune", ".tmp");
            try (OutputStream out = Files.newOutputStream(tmp)) {
                p.store(out, "jitnn autotune decisions: choice = variant");
            }
            Files.move(tmp, f, java.nio.file.StandardCopyOption.REPLACE_EXISTING, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException e) {
            System.err.println("[jitnn] autotune: cannot write " + f + ": " + e);
        }
    }

    /** Forgets the decisions loaded in this JVM (they are read again from the files). */
    public static void clear() {
        DECISIONS.clear();
        PENDING.get().clear();
    }

    static Set<String> pendingKeys() {
        return PENDING.get().keySet();
    }
}
