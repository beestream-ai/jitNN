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

    // Persistent storages: the compiled traces whose device copy is current.
    final Set<CompiledTrace> validOn = Collections.newSetFromMap(new IdentityHashMap<>());

    // Non-persistent outputs: the trace and execution that produced the current device contents.
    CompiledTrace boundTrace;
    long boundExec;
    boolean stale;

    public Storage(DType dtype, int numel, boolean persistent) {
        this.dtype = dtype;
        this.numel = numel;
        this.persistent = persistent;
    }

    public static Storage ofHost(TornadoNativeArray array, DType dtype, boolean persistent) {
        Storage s = new Storage(dtype, array.getSize(), persistent);
        s.array = array;
        s.hostValid = true;
        return s;
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
    }
}
