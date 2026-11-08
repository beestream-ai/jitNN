/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.cpu;

import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Which way to run an operator of one shape is fastest: the CPU backend's own kernel, a native BLAS
 * library's GEMM ({@link Blas}), oneDNN ({@link Dnnl}) or BNNS ({@link Bnns}). The first calls
 * take the available ways in turn, three calls each (the first warms the JIT or builds the
 * library's plan, the best of the other two counts), and the fastest runs the shape from then on.
 * Those first calls fall in the program's warm-up, while the JIT compiles, so after {@value
 * #SETTLE} more calls the ways are timed once more the same way, and that choice is final.
 * {@code -Djitnn.cpu.native.choose=kernel},
 * {@code blas}, {@code onednn} or {@code bnns} takes that way whenever it is available, without
 * timing (tests).
 */
public final class Race {

    /** The ways, in the order {@link #run} takes them. */
    public static final String[] WAYS = { "kernel", "blas", "onednn", "bnns" };
    /** Calls per way in a round of timing: the first warms up, the best of the others counts. */
    private static final int RUNS = 3;
    /** Calls on the first round's choice before the second, final round of timing. */
    static final int SETTLE = 64;
    private static final boolean VERBOSE = Boolean.getBoolean("jitnn.verbose");
    private static final Map<String, Race> RACES = new ConcurrentHashMap<>();

    private final String key;
    private final long[] best = { Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE };
    private final int[] runs = new int[WAYS.length];
    private final boolean[] failed = new boolean[WAYS.length];
    private int decided = -1;
    /** 1 while timing the first round or running its choice, 2 from the second round on. */
    private int round = 1;
    /** Calls since the first round's choice. */
    private int since;

    private Race(String key) {
        this.key = key;
    }

    /**
     * Runs {@code key} (an operator and its shape) one of the ways: {@code kernel}, {@code blas} or
     * {@code onednn}, null when that way is not available. Only the kernel inside a parallel loop
     * (the libraries' threads would oversubscribe the cores).
     */
    public static void run(String key, Runnable kernel, Runnable blas, Runnable onednn) {
        run(key, kernel, blas, onednn, null);
    }

    /** {@link #run(String, Runnable, Runnable, Runnable)} with BNNS ({@code bnns}, null when not available) too. */
    public static void run(String key, Runnable kernel, Runnable blas, Runnable onednn, Runnable bnns) {
        boolean in = Cpu.inside();
        Runnable[] ways = { kernel, in ? null : blas, in ? null : onednn, in ? null : bnns };
        if (ways[1] == null && ways[2] == null && ways[3] == null) {
            kernel.run();
            return;
        }
        Race race = RACES.computeIfAbsent(key, Race::new);
        int way = race.pick(ways);
        long t0 = System.nanoTime();
        try {
            ways[way].run();
        } catch (RuntimeException e) {
            if (way == 0) {
                throw e;
            }
            // the library rejected this shape: the kernel runs it from now on
            race.failed(way, e);
            kernel.run();
            return;
        }
        race.ran(way, System.nanoTime() - t0);
    }

    private synchronized void failed(int way, RuntimeException e) {
        failed[way] = true;
        if (VERBOSE) {
            System.err.println("[jitnn] " + key + " does not run on " + WAYS[way] + ": " + e.getMessage());
        }
    }

    private synchronized int pick(Runnable[] ways) {
        for (int i = 1; i < WAYS.length; i++) {
            if (failed[i]) {
                ways[i] = null;
            }
        }
        String choose = System.getProperty("jitnn.cpu.native.choose", "measure");
        for (int i = 0; i < WAYS.length; i++) {
            if (WAYS[i].equals(choose)) {
                return ways[i] != null ? i : 0;
            }
        }
        if (decided >= 0 && round == 1 && since >= SETTLE) {
            // the program is past its warm-up: time the ways again
            round = 2;
            decided = -1;
            java.util.Arrays.fill(runs, 0);
            java.util.Arrays.fill(best, Long.MAX_VALUE);
        }
        if (decided >= 0 && ways[decided] != null) {
            return decided;
        }
        // the way with the fewest timed calls, so the ways alternate
        int next = -1;
        for (int i = 0; i < WAYS.length; i++) {
            if (ways[i] != null && runs[i] < RUNS && (next < 0 || runs[i] < runs[next])) {
                next = i;
            }
        }
        if (next >= 0) {
            return next;
        }
        // all timed: the fastest available
        int fastest = 0;
        for (int i = 1; i < WAYS.length; i++) {
            if (ways[i] != null && best[i] < best[fastest]) {
                fastest = i;
            }
        }
        decided = fastest;
        since = 0;
        if (VERBOSE) {
            StringBuilder s = new StringBuilder();
            for (int i = 0; i < WAYS.length; i++) {
                if (ways[i] != null) {
                    s.append(String.format(Locale.ROOT, ", %s %.3f ms", WAYS[i], best[i] / 1e6));
                }
            }
            System.err.println("[jitnn] " + key + (round == 1 ? " runs on " : " settles on ") + WAYS[fastest] + s);
        }
        return fastest;
    }

    private synchronized void ran(int way, long nanos) {
        if (decided >= 0) {
            since++;
            return;
        }
        if (runs[way]++ > 0) {
            best[way] = Math.min(best[way], nanos);
        }
    }

    /** Forgets all decisions (tests). */
    public static void reset() {
        RACES.clear();
    }
}
