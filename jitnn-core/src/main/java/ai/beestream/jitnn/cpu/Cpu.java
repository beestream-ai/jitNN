package ai.beestream.jitnn.cpu;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.LockSupport;

import jdk.incubator.vector.FloatVector;

/**
 * The CPU backend's threads. One process-wide pool of {@code jitnn.cpu.threads} threads (default:
 * the physical cores this process may use, at most one socket's: hyper-threads do not speed up
 * FMA-bound kernels, and on two sockets one socket's cores ran ResNet-50 and BERT faster than both;
 * on Apple silicon the performance cores only) runs
 * {@link #parallelFor} in static contiguous chunks, the calling thread taking the first one. A call
 * made while another thread holds the pool, or from inside a parallel loop, runs on the calling
 * thread alone rather than oversubscribing the cores. Idle pool threads, and a caller waiting for
 * them, spin for {@code jitnn.cpu.spinMicros} (default 200, 0 on macOS) before they sleep, so the
 * operators of one step hand work on without waking threads up.
 */
public final class Cpu {

    private static final int THREADS = Math.max(1, Integer.getInteger("jitnn.cpu.threads", defaultThreads()));
    private static final ReentrantLock POOL = new ReentrantLock();
    private static final ThreadLocal<Boolean> INSIDE = ThreadLocal.withInitial(() -> Boolean.FALSE);
    private static volatile Worker[] workers;
    /**
     * 200 us by default, none on macOS: there spinning pool threads made ResNet-50 b1 29% slower on
     * the backend's own kernels and BERT b1 16% slower with Accelerate (M4 Pro).
     */
    private static final long SPIN_NANOS = Long.getLong("jitnn.cpu.spinMicros", System.getProperty("os.name", "").startsWith("Mac") ? 0 : 200) * 1000;

    private Cpu() {
    }

    /** A loop body over the index range [from, to). */
    @FunctionalInterface
    public interface Range {
        void run(int from, int to);
    }

    /** True on a pool thread, or inside a parallel loop's chunk. */
    public static boolean inside() {
        return INSIDE.get();
    }

    /** Threads parallel loops use, the calling thread included. */
    public static int threads() {
        return THREADS;
    }

    /** FP32 lanes in the preferred vector shape (16 on AVX-512, 8 on AVX2, 4 on NEON). */
    public static int lanes() {
        return FloatVector.SPECIES_PREFERRED.length();
    }

    /**
     * Runs {@code body} over [0, n) in at most {@link #threads()} contiguous chunks of at least
     * {@code grain} indices each, and returns when all have run. An exception in any chunk is
     * rethrown on the calling thread.
     */
    public static void parallelFor(int n, int grain, Range body) {
        int chunks = (int) Math.min(THREADS, Math.max(1, n / (long) Math.max(1, grain)));
        if (chunks <= 1 || INSIDE.get() || !POOL.tryLock()) {
            body.run(0, n);
            return;
        }
        try {
            Worker[] ws = workers();
            Job job = new Job(body, n, chunks);
            for (int c = 1; c < chunks; c++) {
                ws[c - 1].submit(job, c);
            }
            INSIDE.set(Boolean.TRUE);
            try {
                job.runChunk(0);
            } finally {
                INSIDE.set(Boolean.FALSE);
            }
            job.await();
        } finally {
            POOL.unlock();
        }
    }

    private static Worker[] workers() {
        Worker[] ws = workers;
        if (ws == null) {
            ws = new Worker[THREADS - 1];
            for (int i = 0; i < ws.length; i++) {
                ws[i] = new Worker("jitnn-cpu-" + (i + 1));
                ws[i].start();
            }
            workers = ws;
        }
        return ws;
    }

    /** One parallel loop: its chunks and how many are still running. */
    private static final class Job {
        final Range body;
        final int n;
        final int chunks;
        final Thread caller = Thread.currentThread();
        final java.util.concurrent.atomic.AtomicInteger pending;
        volatile Throwable failure;

        Job(Range body, int n, int chunks) {
            this.body = body;
            this.n = n;
            this.chunks = chunks;
            this.pending = new java.util.concurrent.atomic.AtomicInteger(chunks);
        }

        void runChunk(int c) {
            int from = (int) ((long) n * c / chunks);
            int to = (int) ((long) n * (c + 1) / chunks);
            try {
                body.run(from, to);
            } catch (Throwable t) {
                failure = t;
            } finally {
                if (pending.decrementAndGet() == 0) {
                    LockSupport.unpark(caller);
                }
            }
        }

        void await() {
            long deadline = System.nanoTime() + SPIN_NANOS;
            while (pending.get() != 0) {
                if (System.nanoTime() < deadline) {
                    Thread.onSpinWait();
                } else {
                    LockSupport.park(this);
                }
            }
            Throwable t = failure;
            if (t instanceof RuntimeException r) {
                throw r;
            }
            if (t instanceof Error e) {
                throw e;
            }
            if (t != null) {
                throw new IllegalStateException(t);
            }
        }
    }

    /** A pool thread: runs the chunk it is given, then parks until the next one. */
    private static final class Worker extends Thread {
        private volatile Job job;
        private volatile int chunk;
        private volatile boolean parked;

        Worker(String name) {
            super(name);
            setDaemon(true);
        }

        void submit(Job j, int c) {
            chunk = c;
            job = j;
            if (parked) {
                LockSupport.unpark(this);
            }
        }

        @Override
        public void run() {
            INSIDE.set(Boolean.TRUE);
            long deadline = System.nanoTime() + SPIN_NANOS;
            while (true) {
                Job j = job;
                if (j == null) {
                    if (System.nanoTime() < deadline) {
                        Thread.onSpinWait();
                        continue;
                    }
                    // announce the sleep, then look once more: a job submitted in between is seen
                    parked = true;
                    if (job == null) {
                        LockSupport.park(this);
                    }
                    parked = false;
                    continue;
                }
                job = null;
                j.runChunk(chunk);
                deadline = System.nanoTime() + SPIN_NANOS;
            }
        }
    }

    /**
     * Physical cores among the processors this process may use (the available processors divided by
     * the hardware threads per core), at most the cores of one socket, or the performance cores on
     * Apple silicon; the available processors when
     * the topology is unknown (Linux sysfs, macOS sysctl).
     */
    static int defaultThreads() {
        int logical = Runtime.getRuntime().availableProcessors();
        int[] t = topology();
        return Math.max(1, Math.min(logical / Math.max(1, t[0]), t[1] > 0 ? t[1] : Integer.MAX_VALUE));
    }

    /** {hardware threads per core, cores per socket (0 when unknown)}. */
    private static int[] topology() {
        try {
            Path base = Path.of("/sys/devices/system/cpu");
            if (Files.isDirectory(base)) {
                Set<String> cores = new HashSet<>();
                Set<String> packages = new HashSet<>();
                int cpus = 0;
                try (var dirs = Files.newDirectoryStream(base, "cpu[0-9]*")) {
                    for (Path d : dirs) {
                        Path core = d.resolve("topology/core_id");
                        Path pkg = d.resolve("topology/physical_package_id");
                        if (Files.isReadable(core) && Files.isReadable(pkg)) {
                            String p = Files.readString(pkg).trim();
                            cores.add(p + ":" + Files.readString(core).trim());
                            packages.add(p);
                            cpus++;
                        }
                    }
                }
                if (cores.isEmpty()) {
                    return new int[] { 1, 0 };
                }
                return new int[] { Math.max(1, cpus / cores.size()), cores.size() / packages.size() };
            }
            if (System.getProperty("os.name", "").startsWith("Mac")) {
                int logical = sysctl("hw.logicalcpu");
                int physical = sysctl("hw.physicalcpu");
                // only the performance cores on Apple silicon: on an M4 Pro 8 threads ran ResNet-50 3-19% faster than all 12 cores
                int performance = sysctl("hw.perflevel0.physicalcpu");
                return new int[] { physical > 0 && logical > 0 ? Math.max(1, logical / physical) : 1, performance > 0 ? performance : physical };
            }
        } catch (Exception e) {
            // unknown topology: one thread per core
        }
        return new int[] { 1, 0 };
    }

    /** A numeric sysctl value, 0 when the name is unknown (perflevel names exist only on Apple silicon). */
    private static int sysctl(String name) throws Exception {
        Process p = new ProcessBuilder("sysctl", "-n", name).redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes()).trim();
        p.waitFor();
        try {
            return Integer.parseInt(out);
        } catch (NumberFormatException e) {
            return 0;
        }
    }
}
