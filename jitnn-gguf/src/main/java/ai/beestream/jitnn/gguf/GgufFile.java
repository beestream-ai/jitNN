package ai.beestream.jitnn.gguf;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.channels.FileChannel.MapMode;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ai.beestream.jitnn.DType;
import ai.beestream.jitnn.Tensor;

import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.HalfFloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;
import uk.ac.manchester.tornado.api.types.arrays.ShortArray;
import uk.ac.manchester.tornado.api.types.arrays.TornadoNativeArray;

/**
 * A GGUF file (llama.cpp's model format, versions 2 and 3): its metadata and its tensors, which load
 * as persistent jitNN tensors. The block-quantised weights (Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K, Q3_K) stay
 * in their GGUF blocks and run through {@link ai.beestream.jitnn.ops.Gguf}; F32, F16, BF16 and I32
 * tensors load as those types.
 *
 * <p>Tensors are memory-mapped rather than read: each one is mapped copy-on-write together with the
 * few bytes before it, which become the TornadoVM array header, so loading a model costs no copy and
 * no Java heap, and the file itself is never changed. Where that is not possible (a file that cannot
 * be opened for writing, or data not 8-byte aligned) the tensor is copied instead.
 *
 * <pre>{@code
 * GgufFile f = GgufFile.open(Path.of("model.gguf"));
 * int layers = f.integer("llama.block_count");
 * Tensor wq = f.tensor("blk.0.attn_q.weight");   // Q4_K, shape [out, in]
 * Tensor y = Gguf.linear(x, wq, null);
 * }</pre>
 *
 * <p>Metadata values are Java objects: the 8-, 16- and 32-bit signed and unsigned integers except
 * uint32 are {@code Integer}, uint32 and the 64-bit integers are {@code Long} (a uint64 above
 * {@code Long.MAX_VALUE} reads as negative), then {@code Float}, {@code Double}, {@code Boolean} and
 * {@code String}. Arrays are {@code int[]}, {@code long[]}, {@code float[]}, {@code double[]},
 * {@code boolean[]} and {@code String[]} by the same rule, and {@code Object[]} for arrays of arrays.
 */
public final class GgufFile {

    private static final int MAGIC = 0x46554747; // "GGUF" read little-endian
    private static final int DEFAULT_ALIGNMENT = 32;

    private static final ValueLayout.OfByte U8 = ValueLayout.JAVA_BYTE;
    private static final ValueLayout.OfShort U16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt U32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong U64 = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfDouble F64 = ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final Path path;
    private final int version;
    private final long fileSize;
    private final long dataOffset;
    private final int alignment;
    private final Map<String, Object> metadata;
    private final Map<String, GgufTensorInfo> tensors;

    private GgufFile(Path path, int version, long fileSize, long dataOffset, int alignment, Map<String, Object> metadata, Map<String, GgufTensorInfo> tensors) {
        this.path = path;
        this.version = version;
        this.fileSize = fileSize;
        this.dataOffset = dataOffset;
        this.alignment = alignment;
        this.metadata = Collections.unmodifiableMap(metadata);
        this.tensors = Collections.unmodifiableMap(tensors);
    }

    /** Reads the metadata and the tensor list of {@code path}; tensors are mapped when loaded. */
    public static GgufFile open(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ); Arena arena = Arena.ofConfined()) {
            long size = ch.size();
            MemorySegment file = ch.map(MapMode.READ_ONLY, 0, size, arena);
            Reader r = new Reader(file, path);
            if (size < 24 || r.u32() != MAGIC) {
                throw new IOException(path + ": not a GGUF file");
            }
            int version = r.u32();
            if (version != 2 && version != 3) {
                throw new IOException(path + ": GGUF version " + Integer.toUnsignedString(version) + " is not supported (2 and 3 are; big-endian files are not)");
            }
            long tensorCount = r.count("tensor count");
            long kvCount = r.count("metadata count");
            Map<String, Object> metadata = new LinkedHashMap<>();
            for (long i = 0; i < kvCount; i++) {
                String key = r.string();
                metadata.put(key, r.value(r.u32()));
            }
            Object a = metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
            int alignment = a instanceof Number n ? (int) n.longValue() : -1;
            if (alignment <= 0 || Integer.bitCount(alignment) != 1) {
                throw new IOException(path + ": general.alignment " + a + " is not a power of two");
            }
            List<long[]> raw = new ArrayList<>();
            List<String> names = new ArrayList<>();
            for (long i = 0; i < tensorCount; i++) {
                names.add(r.string());
                int dims = r.u32();
                if (dims < 0 || dims > 8) {
                    throw new IOException(path + ": tensor " + names.getLast() + " has " + dims + " dimensions");
                }
                long[] t = new long[dims + 2];
                for (int d = 0; d < dims; d++) {
                    t[d] = r.u64();
                }
                t[dims] = Integer.toUnsignedLong(r.u32());
                t[dims + 1] = r.u64();
                raw.add(t);
            }
            long dataOffset = alignUp(r.pos, alignment);
            Map<String, GgufTensorInfo> tensors = new LinkedHashMap<>();
            for (int i = 0; i < names.size(); i++) {
                long[] t = raw.get(i);
                int dims = t.length - 2;
                long[] shape = new long[dims];
                for (int d = 0; d < dims; d++) {
                    shape[d] = t[dims - 1 - d]; // GGUF lists the innermost axis first
                }
                int typeId = (int) t[dims];
                GgmlType type = GgmlType.fromId(typeId);
                long offset = dataOffset + t[dims + 1];
                long numel = 1;
                for (long d : shape) {
                    numel = Math.multiplyExact(numel, d);
                }
                long bytes = type == null ? -1 : type.byteSize(numel);
                if (t[dims + 1] % alignment != 0 || bytes >= 0 && offset + bytes > size) {
                    throw new IOException(path + ": tensor " + names.get(i) + " lies outside the file or is not aligned");
                }
                tensors.put(names.get(i), new GgufTensorInfo(names.get(i), shape, typeId, type, offset, bytes));
            }
            return new GgufFile(path, version, size, dataOffset, alignment, metadata, tensors);
        } catch (IndexOutOfBoundsException e) {
            throw new IOException(path + ": truncated GGUF file", e);
        }
    }

    private static long alignUp(long x, long a) {
        return (x + a - 1) / a * a;
    }

    // ---------------------------------------------------------------- file

    /** The file this was read from. */
    public Path path() {
        return path;
    }

    /** The GGUF format version (2 or 3). */
    public int version() {
        return version;
    }

    /** The data alignment in bytes ({@code general.alignment}, 32 by default). */
    public int alignment() {
        return alignment;
    }

    /** Byte offset of the tensor data section. */
    public long dataOffset() {
        return dataOffset;
    }

    // ---------------------------------------------------------------- metadata

    /** All metadata, in file order. */
    public Map<String, Object> metadata() {
        return metadata;
    }

    /** The metadata keys, in file order. */
    public Set<String> keys() {
        return metadata.keySet();
    }

    /** Whether the metadata has {@code key}. */
    public boolean has(String key) {
        return metadata.containsKey(key);
    }

    /** The value of {@code key}, or {@code null} when absent. */
    public Object get(String key) {
        return metadata.get(key);
    }

    /** {@code general.architecture}, for example {@code llama}, or {@code null} when absent. */
    public String architecture() {
        return metadata.get("general.architecture") instanceof String s ? s : null;
    }

    /** A string value; throws when the key is absent or holds another type. */
    public String string(String key) {
        return required(key, String.class);
    }

    /** An integer value of any width that fits in an {@code int}. */
    public int integer(String key) {
        long v = required(key, Number.class).longValue();
        if (!(get(key) instanceof Integer || get(key) instanceof Long) || v != (int) v) {
            throw new IllegalArgumentException(key + " = " + get(key) + " is not an int");
        }
        return (int) v;
    }

    /** {@link #integer(String)}, or {@code fallback} when the key is absent. */
    public int integer(String key, int fallback) {
        return has(key) ? integer(key) : fallback;
    }

    /** An integer value of any width. */
    public long longValue(String key) {
        Number n = required(key, Number.class);
        if (!(n instanceof Integer || n instanceof Long)) {
            throw new IllegalArgumentException(key + " = " + n + " is not an integer");
        }
        return n.longValue();
    }

    /** A float32 or float64 value (an integer value is accepted too). */
    public float floatValue(String key) {
        return required(key, Number.class).floatValue();
    }

    /** {@link #floatValue(String)}, or {@code fallback} when the key is absent. */
    public float floatValue(String key, float fallback) {
        return has(key) ? floatValue(key) : fallback;
    }

    /** A boolean value; throws when the key is absent or holds another type. */
    public boolean bool(String key) {
        return required(key, Boolean.class);
    }

    /**
     * An array of strings (a tokenizer's vocabulary, say); throws when the key is absent or holds another
     * type.
     */
    public String[] strings(String key) {
        return required(key, String[].class);
    }

    /** An array of float32 values; throws when the key is absent or holds another type. */
    public float[] floats(String key) {
        return required(key, float[].class);
    }

    /** An array of 8-, 16- or 32-bit integers (other than uint32). */
    public int[] ints(String key) {
        return required(key, int[].class);
    }

    /** An array of uint32 or 64-bit integers. */
    public long[] longs(String key) {
        return required(key, long[].class);
    }

    private <T> T required(String key, Class<T> type) {
        Object v = metadata.get(key);
        if (v == null) {
            throw new IllegalArgumentException(path + ": no metadata key " + key);
        }
        if (!type.isInstance(v)) {
            throw new IllegalArgumentException(key + " is " + v.getClass().getSimpleName() + ", not " + type.getSimpleName());
        }
        return type.cast(v);
    }

    // ---------------------------------------------------------------- tensors

    /** The tensors listed in the file, by name, in file order. */
    public Map<String, GgufTensorInfo> tensorInfos() {
        return tensors;
    }

    /** Whether the file lists a tensor named {@code name}. */
    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    /** The listing of tensor {@code name}: its type, shape and offset; throws when there is none. */
    public GgufTensorInfo info(String name) {
        GgufTensorInfo t = tensors.get(name);
        if (t == null) {
            throw new IllegalArgumentException(path + ": no tensor " + name);
        }
        return t;
    }

    /** Tensor {@code name} as a persistent jitNN tensor, memory-mapped where possible (see the class comment). */
    public Tensor tensor(String name) throws IOException {
        return tensor(name, false);
    }

    /** Tensor {@code name}; {@code copy} reads it into memory instead of mapping it. */
    public Tensor tensor(String name, boolean copy) throws IOException {
        GgufTensorInfo t = info(name);
        DType dtype = dtypeOf(t);
        int[] dims = new int[t.shape().length];
        for (int i = 0; i < dims.length; i++) {
            dims[i] = (int) t.shape()[i];
        }
        TornadoNativeArray array = copy ? null : mapped(t, dtype);
        if (array == null) {
            array = copied(t, dtype);
        }
        return Tensor.persistent(array, dtype, dims);
    }

    /** Every tensor in the file (all must be of a type jitNN loads), by name, in file order. */
    public Map<String, Tensor> tensors() throws IOException {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (String name : tensors.keySet()) {
            out.put(name, tensor(name));
        }
        return out;
    }

    /** The raw bytes of tensor {@code name}, mapped read-only and released when unreachable. */
    public MemorySegment raw(String name) throws IOException {
        GgufTensorInfo t = info(name);
        if (t.bytes() < 0) {
            throw new UnsupportedOperationException(name + ": ggml type " + t.typeId() + " is unknown, so its size is too");
        }
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            return ch.map(MapMode.READ_ONLY, t.offset(), t.bytes(), Arena.ofAuto());
        }
    }

    private static DType dtypeOf(GgufTensorInfo t) {
        if (t.type() == null || t.type().dtype() == null) {
            throw new UnsupportedOperationException(t.name() + ": ggml type " + (t.type() == null ? "id " + t.typeId() : t.type()) + " has no jitNN tensor type");
        }
        if (t.numel() > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(t.name() + ": " + t.numel() + " values; jitNN tensors hold at most " + Integer.MAX_VALUE);
        }
        return t.type().dtype();
    }

    /**
     * The tensor mapped copy-on-write with the {@code ARRAY_HEADER} bytes before it as the array
     * header, or {@code null} when that cannot be done here.
     */
    private TornadoNativeArray mapped(GgufTensorInfo t, DType dtype) {
        long header = TornadoNativeArray.ARRAY_HEADER;
        if (t.offset() < header || t.offset() % 8 != 0) {
            return null;
        }
        MemorySegment seg;
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            // PRIVATE: writes (the header) go to private copies of the pages, never to the file
            seg = ch.map(MapMode.PRIVATE, t.offset() - header, t.bytes() + header, Arena.ofAuto());
        } catch (IOException | UnsupportedOperationException | SecurityException e) {
            return null; // read-only file or file system: copy instead
        }
        seg.asSlice(0, header).fill((byte) 0);
        return switch (dtype) {
            case F32 -> FloatArray.fromSegmentShallow(seg);
            case F16 -> HalfFloatArray.fromSegmentShallow(seg);
            case BF16 -> ShortArray.fromSegmentShallow(seg);
            case I32 -> IntArray.fromSegmentShallow(seg);
            default -> ByteArray.fromSegmentShallow(seg);
        };
    }

    private TornadoNativeArray copied(GgufTensorInfo t, DType dtype) throws IOException {
        long n = t.numel();
        TornadoNativeArray array = switch (dtype) {
            case F32 -> new FloatArray(n);
            case F16 -> new HalfFloatArray(n);
            case BF16 -> new ShortArray(n);
            case I32 -> new IntArray(n);
            default -> new ByteArray(t.bytes());
        };
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ); Arena arena = Arena.ofConfined()) {
            MemorySegment src = ch.map(MapMode.READ_ONLY, t.offset(), t.bytes(), arena);
            MemorySegment.copy(src, 0, array.getSegment(), 0, t.bytes());
        }
        return array;
    }

    @Override
    public String toString() {
        return "GgufFile[" + path + ", v" + version + ", " + metadata.size() + " keys, " + tensors.size() + " tensors, " + fileSize + " bytes]";
    }

    // ---------------------------------------------------------------- parsing

    /** Sequential little-endian reads over the mapped file. */
    private static final class Reader {
        private final MemorySegment s;
        private final Path path;
        long pos;

        Reader(MemorySegment s, Path path) {
            this.s = s;
            this.path = path;
        }

        int u8() {
            return s.get(U8, pos++) & 0xff;
        }

        int u16() {
            int v = s.get(U16, pos) & 0xffff;
            pos += 2;
            return v;
        }

        int u32() {
            int v = s.get(U32, pos);
            pos += 4;
            return v;
        }

        long u64() {
            long v = s.get(U64, pos);
            pos += 8;
            return v;
        }

        /** A uint64 count that must be plausible for the bytes left. */
        long count(String what) throws IOException {
            long n = u64();
            if (n < 0 || n > s.byteSize() - pos) {
                throw new IOException(path + ": " + what + " " + Long.toUnsignedString(n) + " exceeds the file");
            }
            return n;
        }

        String string() throws IOException {
            long n = count("string length");
            if (n > Integer.MAX_VALUE) {
                throw new IOException(path + ": string of " + n + " bytes");
            }
            byte[] b = s.asSlice(pos, n).toArray(U8);
            pos += n;
            return new String(b, StandardCharsets.UTF_8);
        }

        Object value(int type) throws IOException {
            return switch (type) {
                case 0 -> u8();                           // uint8
                case 1 -> (int) (byte) u8();              // int8
                case 2 -> u16();                          // uint16
                case 3 -> (int) (short) u16();            // int16
                case 4 -> Integer.toUnsignedLong(u32());  // uint32
                case 5 -> u32();                          // int32
                case 6 -> Float.intBitsToFloat(u32());    // float32
                case 7 -> u8() != 0;                      // bool
                case 8 -> string();
                case 9 -> array();
                case 10, 11 -> u64();                     // uint64, int64
                case 12 -> Double.longBitsToDouble(u64());
                default -> throw new IOException(path + ": unknown metadata value type " + type + " at byte " + (pos - 4));
            };
        }

        Object array() throws IOException {
            int type = u32();
            long n = count("array length");
            if (n > Integer.MAX_VALUE) {
                throw new IOException(path + ": array of " + n + " values");
            }
            int len = (int) n;
            switch (type) {
                case 0, 1, 2, 3, 5 -> {
                    int[] a = new int[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = (Integer) value(type);
                    }
                    return a;
                }
                case 4, 10, 11 -> {
                    long[] a = new long[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = (Long) value(type);
                    }
                    return a;
                }
                case 6 -> {
                    float[] a = new float[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = Float.intBitsToFloat(u32());
                    }
                    return a;
                }
                case 12 -> {
                    double[] a = new double[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = Double.longBitsToDouble(u64());
                    }
                    return a;
                }
                case 7 -> {
                    boolean[] a = new boolean[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = u8() != 0;
                    }
                    return a;
                }
                case 8 -> {
                    String[] a = new String[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = string();
                    }
                    return a;
                }
                case 9 -> {
                    Object[] a = new Object[len];
                    for (int i = 0; i < len; i++) {
                        a[i] = array();
                    }
                    return a;
                }
                default -> throw new IOException(path + ": unknown array element type " + type);
            }
        }
    }
}
