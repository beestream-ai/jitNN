package ai.beestream.jitnn.io;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.engine.Storage;

import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * Named tensors and string values: the state of a model, an optimiser or a whole training run,
 * as PyTorch's {@code state_dict}. Built from live objects ({@code Module.stateDict()},
 * {@code Optimizer.stateDict()}) it refers to their tensors, read when it is saved; loaded from a
 * file, it reads each tensor only when it is copied in. Nested parts are flattened with dotted
 * names ({@code model.parameters.0}). Files are safetensors (see {@link SafeTensors}).
 *
 * <pre>{@code
 * StateDict run = new StateDict().put("model", model.stateDict()).put("optimizer", opt.stateDict());
 * run.save(Path.of("step-1000.safetensors"));
 *
 * StateDict in = StateDict.load(Path.of("step-1000.safetensors"));
 * model.loadStateDict(in.sub("model"));
 * opt.loadStateDict(in.sub("optimizer"));
 * }</pre>
 */
public final class StateDict {

    private final Map<String, Tensor> live = new LinkedHashMap<>();
    private final Map<String, SafeTensors.Entry> stored = new LinkedHashMap<>();
    private final Map<String, String> values = new LinkedHashMap<>();
    private final Path file;

    /** An empty dictionary, to fill with {@code put} and save. */
    public StateDict() {
        this(null);
    }

    private StateDict(Path file) {
        this.file = file;
    }

    /** Adds a tensor; its contents are read when the dictionary is saved. */
    public StateDict put(String name, Tensor t) {
        checkFree(name);
        SafeTensors.name(t.dtype());
        live.put(name, t);
        return this;
    }

    /** Adds a value. */
    public StateDict put(String name, String value) {
        checkFree(name);
        values.put(name, value);
        return this;
    }

    /** Adds an integer value (stored as its decimal string). */
    public StateDict put(String name, long value) {
        return put(name, Long.toString(value));
    }

    /** Adds a float value (stored as its decimal string). */
    public StateDict put(String name, float value) {
        return put(name, Float.toString(value));
    }

    /** Adds every entry of {@code part} under {@code prefix.}. */
    public StateDict put(String prefix, StateDict part) {
        if (!part.stored.isEmpty() && part.file != file) {
            throw new IllegalArgumentException("entries loaded from a file can only be added back to a dictionary of that file");
        }
        part.live.forEach((k, v) -> put(prefix + "." + k, v));
        part.values.forEach((k, v) -> put(prefix + "." + k, v));
        part.stored.forEach((k, v) -> {
            checkFree(prefix + "." + k);
            stored.put(prefix + "." + k, v);
        });
        return this;
    }

    /** The entries under {@code prefix.}, with the prefix removed. */
    public StateDict sub(String prefix) {
        String p = prefix + ".";
        StateDict s = new StateDict(file);
        live.forEach((k, v) -> {
            if (k.startsWith(p)) {
                s.live.put(k.substring(p.length()), v);
            }
        });
        stored.forEach((k, v) -> {
            if (k.startsWith(p)) {
                s.stored.put(k.substring(p.length()), v);
            }
        });
        values.forEach((k, v) -> {
            if (k.startsWith(p)) {
                s.values.put(k.substring(p.length()), v);
            }
        });
        return s;
    }

    /** The tensor names, in insertion (or file) order. */
    public Set<String> tensorNames() {
        Set<String> names = new LinkedHashSet<>(live.keySet());
        names.addAll(stored.keySet());
        return names;
    }

    /** The value names. */
    public Set<String> valueNames() {
        return values.keySet();
    }

    /** Whether the dictionary holds a tensor named {@code name}. */
    public boolean hasTensor(String name) {
        return live.containsKey(name) || stored.containsKey(name);
    }

    /** The value {@code name}; throws when it is missing. */
    public String value(String name) {
        String v = values.get(name);
        if (v == null) {
            throw new IllegalArgumentException("no value " + name + " (has " + values.keySet() + ")");
        }
        return v;
    }

    /** The value {@code name} as a long; throws when it is missing. */
    public long longValue(String name) {
        return Long.parseLong(value(name));
    }

    /** The value {@code name} as an int; throws when it is missing. */
    public int intValue(String name) {
        return Integer.parseInt(value(name));
    }

    /** The value {@code name} as a float; throws when it is missing. */
    public float floatValue(String name) {
        return Float.parseFloat(value(name));
    }

    /** The dtype of tensor {@code name}. */
    public DType dtype(String name) {
        Tensor t = live.get(name);
        return t != null ? t.dtype() : entry(name).dtype();
    }

    /** The shape of tensor {@code name}. */
    public int[] shape(String name) {
        Tensor t = live.get(name);
        if (t != null) {
            return t.shape().dims();
        }
        long[] s = entry(name).shape();
        int[] dims = new int[s.length];
        for (int i = 0; i < s.length; i++) {
            dims[i] = Math.toIntExact(s[i]);
        }
        return dims;
    }

    /**
     * Overwrites the persistent tensor {@code target} (a parameter, a buffer, optimiser state) with
     * tensor {@code name}, in place, so plans compiled with {@code target} keep using it. Dtype and
     * shape must match.
     */
    public void copyInto(String name, Tensor target) {
        check(name, target);
        // operators recorded so far may still read or write the target
        Engine.sync();
        Storage s = target.storage();
        MemorySegment dst = s.host().getSegment();
        Tensor source = live.get(name);
        if (source != null) {
            MemorySegment.copy(bytes(source), 0, dst, 0, byteSize(target));
        } else {
            try {
                SafeTensors.read(file, entry(name), dst);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        s.hostWritten();
    }

    /** Throws unless tensor {@code name} can be copied into {@code target} (see {@link #copyInto}). */
    public void check(String name, Tensor target) {
        if (!target.storage().isPersistent()) {
            throw new IllegalArgumentException(name + ": the target must be persistent (a parameter, buffer or optimiser state)");
        }
        if (dtype(name) != target.dtype() || !Arrays.equals(shape(name), target.shape().dims())) {
            throw new IllegalArgumentException(name + ": stored " + dtype(name) + Arrays.toString(shape(name)) + ", target " + target.dtype() + Arrays.toString(target.shape().dims()));
        }
    }

    /** Tensor {@code name} as a new persistent tensor (a weight that stays on the device). */
    public Tensor tensor(String name) {
        DType dtype = dtype(name);
        int[] dims = shape(name);
        long n = 1;
        for (int d : dims) {
            n *= d;
        }
        TornadoNativeArray array = Storage.newArray(dtype, Math.toIntExact(n));
        Tensor t = Tensor.persistent(array, dtype, dims.length == 0 ? new int[0] : dims);
        copyInto(name, t);
        return t;
    }

    /** Writes the dictionary to {@code path} as safetensors; reading live tensors waits for them. */
    public void save(Path path) {
        Map<String, SafeTensors.Source> sources = new LinkedHashMap<>();
        if (!stored.isEmpty()) {
            for (String name : stored.keySet()) {
                Tensor t = tensor(name);
                sources.put(name, source(t));
            }
        }
        live.forEach((k, t) -> sources.put(k, source(t)));
        try {
            SafeTensors.write(path, sources, values);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Opens a safetensors file; tensors are read when copied in. */
    public static StateDict load(Path path) {
        try {
            SafeTensors.Header h = SafeTensors.readHeader(path);
            StateDict d = new StateDict(path);
            d.stored.putAll(h.entries());
            d.values.putAll(h.metadata());
            return d;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static SafeTensors.Source source(Tensor t) {
        int[] dims = t.shape().dims();
        long[] shape = new long[dims.length];
        for (int i = 0; i < dims.length; i++) {
            shape[i] = dims[i];
        }
        return new SafeTensors.Source(t.dtype(), shape, bytes(t));
    }

    private static MemorySegment bytes(Tensor t) {
        Engine.sync(t);
        return t.storage().host().getSegment().asSlice(0, byteSize(t));
    }

    private static long byteSize(Tensor t) {
        return (long) t.numel() * t.dtype().bytes();
    }

    private SafeTensors.Entry entry(String name) {
        SafeTensors.Entry e = stored.get(name);
        if (e == null) {
            throw new IllegalArgumentException("no tensor " + name);
        }
        if (e.dtype() == null) {
            throw new IllegalArgumentException(name + ": dtype " + e.typeName() + " is not supported (F32, F16, BF16, I32, F8_E4M3 and F8_E5M2 are)");
        }
        if (e.bytes() != e.numel() * e.dtype().bytes()) {
            throw new IllegalArgumentException(name + ": " + e.bytes() + " bytes for " + e.numel() + " " + e.dtype() + " values");
        }
        return e;
    }

    private void checkFree(String name) {
        if (live.containsKey(name) || stored.containsKey(name) || values.containsKey(name)) {
            throw new IllegalArgumentException("duplicate entry " + name);
        }
    }
}
