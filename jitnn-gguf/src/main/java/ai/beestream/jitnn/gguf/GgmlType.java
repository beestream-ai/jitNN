package ai.beestream.jitnn.gguf;

import ai.beestream.jitnn.DType;

/**
 * The tensor element types a GGUF file can hold (ggml's {@code ggml_type}), with their ids, block
 * sizes and the jitNN {@link DType} each one loads as. Types without a jitNN counterpart can be
 * listed and read as raw bytes ({@link GgufFile#raw}), but not loaded as tensors.
 */
public enum GgmlType {
    /** 32-bit float. */
    F32(0, 1, 4, DType.F32),
    /** IEEE 16-bit float. */
    F16(1, 1, 2, DType.F16),
    /** 4-bit blocks of 32 values with an FP16 scale. */
    Q4_0(2, 32, 18, DType.Q4_0),
    /** 4-bit blocks of 32 values with an FP16 scale and minimum. */
    Q4_1(3, 32, 20, DType.Q4_1),
    /** 5-bit blocks of 32 values with an FP16 scale. */
    Q5_0(6, 32, 22, DType.Q5_0),
    /** 5-bit blocks of 32 values with an FP16 scale and minimum. */
    Q5_1(7, 32, 24, DType.Q5_1),
    /** 8-bit blocks of 32 values with an FP16 scale. */
    Q8_0(8, 32, 34, DType.Q8_0),
    /** 8-bit blocks of 32 values with a scale and sum; raw bytes only. */
    Q8_1(9, 32, 36, null),
    /** 2-bit k-quant super-blocks of 256 values. */
    Q2_K(10, 256, 84, DType.Q2_K),
    /** 3-bit k-quant super-blocks of 256 values. */
    Q3_K(11, 256, 110, DType.Q3_K),
    /** 4-bit k-quant super-blocks of 256 values. */
    Q4_K(12, 256, 144, DType.Q4_K),
    /** 5-bit k-quant super-blocks of 256 values. */
    Q5_K(13, 256, 176, DType.Q5_K),
    /** 6-bit k-quant super-blocks of 256 values. */
    Q6_K(14, 256, 210, DType.Q6_K),
    /** 8-bit k-quant super-blocks of 256 values (llama.cpp's intermediate type); raw bytes only. */
    Q8_K(15, 256, 292, null),
    /** 8-bit signed integer; raw bytes only. */
    I8(24, 1, 1, null),
    /** 16-bit signed integer; raw bytes only. */
    I16(25, 1, 2, null),
    /** 32-bit signed integer. */
    I32(26, 1, 4, DType.I32),
    /** 64-bit signed integer; raw bytes only. */
    I64(27, 1, 8, null),
    /** 64-bit float; raw bytes only. */
    F64(28, 1, 8, null),
    /** bfloat16. */
    BF16(30, 1, 2, DType.BF16);

    private static final GgmlType[] BY_ID = new GgmlType[64];

    static {
        for (GgmlType t : values()) {
            BY_ID[t.id] = t;
        }
    }

    private final int id;
    private final int blockValues;
    private final int blockBytes;
    private final DType dtype;

    GgmlType(int id, int blockValues, int blockBytes, DType dtype) {
        this.id = id;
        this.blockValues = blockValues;
        this.blockBytes = blockBytes;
        this.dtype = dtype;
    }

    /** The type with ggml id {@code id}, or {@code null} for an id this reader does not know. */
    public static GgmlType fromId(int id) {
        return id >= 0 && id < BY_ID.length ? BY_ID[id] : null;
    }

    /** ggml's numeric id for this type, as stored in the file. */
    public int id() {
        return id;
    }

    /** Values per block (1 for the plain types). */
    public int blockValues() {
        return blockValues;
    }

    /** Bytes per block. */
    public int blockBytes() {
        return blockBytes;
    }

    /** The jitNN type this one loads as, or {@code null} when jitNN has none. */
    public DType dtype() {
        return dtype;
    }

    /** Bytes taken by {@code elements} values ({@code elements} a multiple of the block size). */
    public long byteSize(long elements) {
        if (elements % blockValues != 0) {
            throw new IllegalArgumentException(this + ": " + elements + " values is not a whole number of blocks of " + blockValues);
        }
        return elements / blockValues * blockBytes;
    }
}
