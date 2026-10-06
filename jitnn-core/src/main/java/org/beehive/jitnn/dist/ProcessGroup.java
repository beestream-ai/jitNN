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
 * Runs one rank per device in this process: rank r is a thread whose plans run on devices[r]
 * (see {@link Engine#onDevice}), and all ranks share one {@link Collective}. Several ranks may
 * name the same device.
 */
public final class ProcessGroup {

    private ProcessGroup() {
    }

    /** Runs {@code body(rank, collective)} on every rank and returns the ranks' results in rank order. */
    public static <T> List<T> run(String[] devices, Collective collective, BiFunction<Integer, Collective, T> body) {
        if (collective.worldSize() != devices.length) {
            throw new IllegalArgumentException(devices.length + " devices for a collective of " + collective.worldSize() + " ranks");
        }
        ExecutorService pool = Executors.newFixedThreadPool(devices.length);
        try {
            List<Future<T>> futures = new ArrayList<>();
            for (int r = 0; r < devices.length; r++) {
                int rank = r;
                futures.add(pool.submit(() -> {
                    try {
                        return Engine.onDevice(devices[rank], () -> body.apply(rank, collective));
                    } catch (RuntimeException | Error e) {
                        if (collective instanceof HostCollective hc) {
                            hc.abort();
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
