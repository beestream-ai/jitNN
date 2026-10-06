package org.beehive.jitnn.engine;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import org.beehive.jitnn.DType;

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
    final boolean persistent;

    TornadoNativeArray array;
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
        if (array == null) {
            array = newArray(dtype, numel);
        }
        return array;
    }

    static TornadoNativeArray newArray(DType dtype, int numel) {
        int n = Math.max(numel, 1);
        return switch (dtype) {
            case F32 -> new FloatArray(n);
            case I32 -> new IntArray(n);
            case F16 -> new uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray(n);
            case BF16 -> new uk.ac.manchester.tornado.api.types.arrays.ShortArray(n);
            case Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K -> new uk.ac.manchester.tornado.api.types.arrays.ByteArray(Math.max(n / dtype.blockValues() * dtype.blockBytes(), 1));
            case FP8_E4M3, FP8_E5M2 -> new uk.ac.manchester.tornado.api.types.arrays.ByteArray(n);
        };
    }

    /** True when {@code a} is the array class that holds {@code dtype}. */
    static boolean holds(TornadoNativeArray a, DType dtype) {
        return switch (dtype) {
            case F32 -> a instanceof FloatArray;
            case I32 -> a instanceof IntArray;
            case F16 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
            case BF16 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.ShortArray;
            case Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, FP8_E4M3, FP8_E5M2 -> a instanceof uk.ac.manchester.tornado.api.types.arrays.ByteArray;
        };
    }

    /** The host array with current contents, fetching it from a device if needed. */
    public TornadoNativeArray host() {
        Engine.ensureHost(this);
        return array;
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
