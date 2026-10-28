package org.beehive.jitnn.engine;

import java.lang.ref.Cleaner;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import org.beehive.jitnn.DType;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Host arrays for outputs that outlive their plan's next execution. When a plan runs again, the
 * previous output that was fetched to the host keeps its contents in an array of its own; allocating
 * that array (zero-filled native memory, outside the Java heap, so it never prompts a collection)
 * cost more than the copy. Arrays come back when their tensor is released ({@code Tensor.release})
 * or, failing that, when it is collected, and are kept up to {@code -Djitnn.hostPool.bytes} (default
 * 256 MiB).
 */
final class HostPool {

    private static volatile long cap = Long.getLong("jitnn.hostPool.bytes", 256L << 20);
    private static final Cleaner CLEANER = Cleaner.create();
    private static final Map<Long, ArrayDeque<TornadoNativeArray>> FREE = new HashMap<>();
    private static long bytes;
    private static final long[] STATS = new long[2];

    private HostPool() {
    }

    /** An array for {@code numel} elements of {@code dtype}: a pooled one, or a new one. */
    static TornadoNativeArray take(DType dtype, int numel) {
        synchronized (FREE) {
            ArrayDeque<TornadoNativeArray> q = FREE.get(key(dtype, numel));
            if (q != null && !q.isEmpty()) {
                TornadoNativeArray a = q.pop();
                bytes -= a.getNumBytesOfSegment();
                STATS[0]++;
                return a;
            }
        }
        STATS[1]++;
        return Storage.newArray(dtype, numel);
    }

    /** {@code array} given to {@code owner}, back in the pool when the owner is released or collected. */
    static Lease lease(Storage owner, TornadoNativeArray array) {
        Lease l = new Lease(array, owner.dtype, owner.numel);
        CLEANER.register(owner, l);
        return l;
    }

    /** Keeps up to {@code bytes} pooled, if that is more than now (the CPU backend's intermediates). */
    static void raiseCap(long bytes) {
        cap = Math.max(cap, bytes);
    }

    /** {@code [arrays reused, arrays allocated]} by {@link #take} so far. */
    static long[] stats() {
        synchronized (FREE) {
            return STATS.clone();
        }
    }

    private static void give(TornadoNativeArray a, DType dtype, int numel) {
        synchronized (FREE) {
            long size = a.getNumBytesOfSegment();
            if (bytes + size > cap) {
                return;
            }
            FREE.computeIfAbsent(key(dtype, numel), k -> new ArrayDeque<>()).push(a);
            bytes += size;
        }
    }

    private static long key(DType dtype, int numel) {
        return ((long) dtype.ordinal() << 32) | (numel & 0xffffffffL);
    }

    /** One return of an array to the pool, by release or by the cleaner, whichever comes first; it holds no reference to its owner. */
    static final class Lease implements Runnable {
        private final TornadoNativeArray array;
        private final DType dtype;
        private final int numel;
        private final AtomicBoolean returned = new AtomicBoolean();

        private Lease(TornadoNativeArray array, DType dtype, int numel) {
            this.array = array;
            this.dtype = dtype;
            this.numel = numel;
        }

        @Override
        public void run() {
            if (returned.compareAndSet(false, true)) {
                give(array, dtype, numel);
            }
        }

        /**
         * Hands the array to {@code owner} without passing through the pool: this lease no longer
         * returns it (its owner's cleaner then does nothing); the new lease does. Null if the array was
         * already returned.
         */
        Lease transfer(Storage owner) {
            if (!returned.compareAndSet(false, true)) {
                return null;
            }
            return lease(owner, array);
        }
    }
}
