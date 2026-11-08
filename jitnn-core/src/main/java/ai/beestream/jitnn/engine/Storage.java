package ai.beestream.jitnn.engine;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import ai.beestream.jitnn.DType;

import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * The memory behind one or more tensors (a tensor and its views share a storage), and where its
 * newest contents live.
 *
 * <p>A <em>persistent</em> storage (parameters, optimiser state) outlives a single execution: its
 * newest contents may be on the host, on the device of one or more compiled traces, or both, and
 * the executor moves it between them on demand. A non-persistent storage is either host data the
 * user created, or the output of exactly one execution of one compiled trace.
 */
public final class Storage {

    private static final AtomicLong IDS = new AtomicLong();

    final long id = IDS.incrementAndGet();
    final DType dtype;
    final int numel;
    boolean persistent;
    /** The plan writing this persistent storage on another thread right now (see TornadoExecutor.execute), or null. */
    CompiledTrace writer;
    /** Executions that read this storage as a host input (see TornadoExecutor's promotion of reused inputs). */
    int inputUses;

    TornadoNativeArray array;

    /**
     * A Java array holding the contents instead of {@link #array} (a float[] or int[] shared by
     * {@code Tensor.wrap}): copied straight into a plan's input buffer, and into a native array only
     * when something else needs one.
     */
    Object heap;
    boolean hostValid;
    /**
     * Created inside autograd's backward pass and not handed to the user as a leaf gradient: no
     * user code can hold it, so its buffer may be reused once the trace no longer needs it.
     */
    boolean internal;

    // Persistent storages: the compiled traces whose device copy is current.
    final Set<CompiledTrace> validOn = Collections.newSetFromMap(new IdentityHashMap<>());

    // Non-persistent outputs: the trace and execution that produced the current device contents.
    CompiledTrace boundTrace;
    long boundExec;
    boolean stale;

    /** Incremented whenever the contents change (host writes, executions that write it); caches derived from it compare it. */
    long version;

    /** A persistent storage whose model was closed: a compiled plan may hand its buffer to another storage. */
    boolean released;
    /** The executor's numbering of the trace being run (see TornadoExecutor's plan memory). */
    int memoStamp;
    int memoLocal;
    /** A persistent storage some compiled plan binds (its array is that plan's buffer). */
    boolean planBound;
    /** A non-persistent storage whose tensor was released ({@link #drop}): its contents are gone. */
    boolean dropped;
    /** The pool's claim on {@link #array} when the executor gave this storage an array of its own. */
    HostPool.Lease lease;

    public Storage(DType dtype, int numel, boolean persistent) {
        this.dtype = dtype;
        this.numel = numel;
        this.persistent = persistent;
        this.internal = !persistent && INTERNAL.get();
    }

    public static Storage ofHost(TornadoNativeArray array, DType dtype, boolean persistent) {
        return ofHost(array, dtype, array.getSize(), persistent);
    }

    /** Host storage of {@code numel} elements in {@code array} (block formats hold fewer bytes per element than one). */
    public static Storage ofHost(TornadoNativeArray array, DType dtype, int numel, boolean persistent) {
        Storage s = new Storage(dtype, numel, persistent);
        s.array = array;
        s.hostValid = true;
        return s;
    }

    /** Host storage over a Java float[] or int[] shared with the caller (see {@link #heap}). */
    public static Storage ofHeap(Object data, DType dtype, int numel) {
        if (!(data instanceof float[] && dtype == DType.F32) && !(data instanceof int[] && dtype == DType.I32)) {
            throw new IllegalArgumentException("a shared host array must be float[] (F32) or int[] (I32)");
        }
        Storage s = new Storage(dtype, numel, false);
        s.heap = data;
        s.hostValid = true;
        return s;
    }

    /** The native array, made from the shared Java array on first use. */
    TornadoNativeArray materialize() {
        if (array == null && heap != null) {
            array = newArray(dtype, numel);
            copyHeap(heap, array, numel);
            heap = null;
        }
        return array;
    }

    /** Copies a shared Java array into a native array of the same type in one bulk copy. */
    static void copyHeap(Object heap, TornadoNativeArray to, int numel) {
        if (heap instanceof float[] f) {
            java.lang.foreign.MemorySegment.copy(f, 0, to.getSegment(), java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, numel);
        } else {
            java.lang.foreign.MemorySegment.copy((int[]) heap, 0, to.getSegment(), java.lang.foreign.ValueLayout.JAVA_INT, 0, numel);
        }
    }

    private static final ThreadLocal<Boolean> INTERNAL = ThreadLocal.withInitial(() -> Boolean.FALSE);

    /** Marks storages created on this thread, until the scope closes, as internal (autograd). */
    public static AutoCloseable internalScope() {
        boolean previous = INTERNAL.get();
        INTERNAL.set(Boolean.TRUE);
        return () -> INTERNAL.set(previous);
    }

    static boolean inInternalScope() {
        return INTERNAL.get();
    }

    /** The user can see this storage (a leaf gradient): never reuse its buffer. */
    public void markVisible() {
        internal = false;
    }

    /** True when the host array holds the current contents (reading it needs no sync). */
    public boolean isHostValid() {
        return hostValid;
    }

    public DType dtype() {
        return dtype;
    }

    public int numel() {
        return numel;
    }

    public boolean isPersistent() {
        return persistent;
    }

    public long id() {
        return id;
    }

    /** Changes whenever the contents do. */
    public long version() {
        return version;
    }

    /** Created by an inference forward pass or by autograd, and not handed to the user: rewrites may fuse it away. */
    public boolean isInternal() {
        return internal;
    }

    /** Its producer was fused into a consumer, so it is never computed; reading it is an error. */
    public void markFused() {
        stale = true;
    }

    TornadoNativeArray allocate() {
        materialize();
        if (array == null) {
            array = newArray(dtype, numel);
        }
        return array;
    }

    /** A host array that holds {@code numel} values of {@code dtype}. */
    public static TornadoNativeArray newArray(DType dtype, int numel) {
        int n = Math.max(numel, 1);
        return switch (dtype) {
            case F32 -> new FloatArray(n);
            case I32 -> new IntArray(n);
            case F16 -> new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(n);
            case BF16 -> new uk.ac.manchester.tornado.api.types.arrays.ShortArray(n);
            case Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K, Q3_K -> new uk.ac.manchester.tornado.api.types.arrays.ByteArray(Math.max(n / dtype.blockValues() * dtype.blockBytes(), 1));
            case FP8_E4M3, FP8_E5M2 -> new uk.ac.manchester.tornado.api.types.arrays.ByteArray(n);
        };
    }

    /** True when {@code a} is the array class that holds {@code dtype}. */
    public static boolean holds(TornadoNativeArray a, DType dtype) {
        return switch (dtype) {
            case F32 -> a instanceof FloatArray;
            case I32 -> a instanceof IntArray;
            case F16 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
            case BF16 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.ShortArray;
            case Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K, Q3_K, FP8_E4M3, FP8_E5M2 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.ByteArray;
        };
    }

    /** The host array with current contents, fetching it from a device if needed. */
    public TornadoNativeArray host() {
        Engine.ensureHost(this);
        return materialize();
    }

    /**
     * Declares that the storage is no longer used. A persistent storage (its model was closed): a
     * compiled plan that binds it can then be reused for another model of the same structure without
     * compiling again. Any other storage drops its contents ({@link #drop}). Using the storage
     * afterwards is an error.
     */
    public void release() {
        if (persistent) {
            // work recorded before the model was closed may still read it
            if (Engine.pending() > 0) {
                Engine.sync();
            }
            released = true;
        } else {
            drop();
        }
    }

    /**
     * Drops the contents of a non-persistent storage whose tensors the user no longer needs
     * ({@code Tensor.release}): an output no longer needs copying out of its plan's buffer when the
     * plan runs again, and an array of its own goes back to the pool. Reading it afterwards is an
     * error. Persistent storages (weights) are released with their model instead: nothing happens.
     */
    public void drop() {
        if (persistent || dropped) {
            return;
        }
        // work recorded before the release may still read it
        if (Engine.pending() > 0) {
            Engine.sync();
        }
        dropped = true;
        hostValid = false;
        heap = null;
        array = null;
        if (lease != null) {
            lease.run();
            lease = null;
        }
    }

    /** Marks the host copy as newer than every device copy (after the user wrote it). */
    public void hostWritten() {
        if (stale) {
            throw new IllegalStateException("storage was reused by a later execution");
        }
        hostValid = true;
        validOn.clear();
        boundTrace = null;
        version++;
    }
}
