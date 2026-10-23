package org.beehive.jitnn.dist;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.BiFunction;

import org.beehive.jitnn.engine.Engine;

/**
 * Runs one rank per device in this process: rank firstRank + r is a thread whose plans run on
 * devices[r] (see {@link Engine#onDevice}), and all ranks share one {@link Collective}. Several
 * ranks may name the same device (not with NCCL). With a collective that spans several processes
 * ({@link NcclCollective#NcclCollective(String, int, int, String...)}), each process runs its own
 * ranks.
 */
public final class ProcessGroup {

    private ProcessGroup() {
    }

    /** Runs {@code body(rank, collective)} on every rank and returns the ranks' results in rank order. */
    public static <T> List<T> run(String[] devices, Collective collective, BiFunction<Integer, Collective, T> body) {
        if (collective.localRanks() != devices.length) {
            throw new IllegalArgumentException(devices.length + " devices for a collective of " + collective.localRanks() + " ranks in this process");
        }
        ExecutorService pool = Executors.newFixedThreadPool(devices.length);
        // the failure that made the others abort, reported instead of theirs
        java.util.concurrent.atomic.AtomicReference<Throwable> first = new java.util.concurrent.atomic.AtomicReference<>();
        try {
            List<Future<T>> futures = new ArrayList<>();
            for (int r = 0; r < devices.length; r++) {
                int rank = collective.firstRank() + r;
                String device = devices[r];
                futures.add(pool.submit(() -> {
                    try {
                        return Engine.onDevice(device, () -> body.apply(rank, collective));
                    } catch (RuntimeException | Error e) {
                        if (first.compareAndSet(null, new IllegalStateException("rank " + rank + " on " + device + " failed: " + e, e))) {
                            collective.abort();
                        }
                        throw e;
                    }
                }));
            }
            List<T> out = new ArrayList<>();
            for (Future<T> f : futures) {
                out.add(f.get());
            }
            return out;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (ExecutionException e) {
            Throwable cause = first.get();
            if (cause instanceof IllegalStateException ise) {
                throw ise;
            }
            throw new IllegalStateException("a rank failed: " + e.getCause(), e.getCause());
        } finally {
            pool.shutdownNow();
        }
    }

    /** {@link #run} with a {@link HostCollective}. */
    public static <T> List<T> run(String[] devices, BiFunction<Integer, Collective, T> body) {
        return run(devices, new HostCollective(devices.length), body);
    }
}
