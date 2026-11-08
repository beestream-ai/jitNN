package ai.beestream.jitnn;

/**
 * Element type of a tensor: FP32 data, INT32 labels and indices, and the half-precision storage
 * types for mixed precision, IEEE FP16 and bfloat16 (the top 16 bits of an FP32), and Q8_0, GGUF's
 * 8-bit blocks (per 32 consecutive values an FP16 scale and 32 signed bytes, 34 bytes), and the
 * OCP 8-bit floats FP8 E4M3 and E5M2, one byte per value, stored without their scale (see
 * {@link ai.beestream.jitnn.ops.Fp8}). Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K and Q3_K are GGUF's other block
 * formats, stored as their raw blocks (see {@link ai.beestream.jitnn.ops.Gguf}).
 */
public enum DType {
    /** IEEE 32-bit float, the default. */
    F32(4),
    /** 32-bit signed integer, for labels and indices. */
    I32(4),
    /** IEEE 16-bit float. */
    F16(2),
    /** bfloat16: the top 16 bits of an FP32. */
    BF16(2),
    /** GGUF's 8-bit blocks: per 32 values an FP16 scale and 32 signed bytes (34 bytes). */
    Q8_0(1),
    /** OCP 8-bit float with 4 exponent and 3 mantissa bits, stored without its scale. */
    FP8_E4M3(1),
    /** OCP 8-bit float with 5 exponent and 2 mantissa bits, stored without its scale. */
    FP8_E5M2(1),
    /** GGUF's 4-bit blocks of 32 values with an FP16 scale (18 bytes). */
    Q4_0(1),
    /** GGUF's 4-bit blocks of 32 values with an FP16 scale and minimum (20 bytes). */
    Q4_1(1),
    /** GGUF's 4-bit k-quant super-blocks of 256 values (144 bytes). */
    Q4_K(1),
    /** GGUF's 5-bit k-quant super-blocks of 256 values (176 bytes). */
    Q5_K(1),
    /** GGUF's 6-bit k-quant super-blocks of 256 values (210 bytes). */
    Q6_K(1),
    /** GGUF's 5-bit blocks of 32 values with an FP16 scale (22 bytes). */
    Q5_0(1),
    /** GGUF's 5-bit blocks of 32 values with an FP16 scale and minimum (24 bytes). */
    Q5_1(1),
    /** GGUF's 2-bit k-quant super-blocks of 256 values (84 bytes). */
    Q2_K(1),
    /** GGUF's 3-bit k-quant super-blocks of 256 values (110 bytes). */
    Q3_K(1);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    /**
     * Bytes per element: 4, 2, or 1 for the 8-bit and block formats (whose blocks {@link #blockBytes}
     * sizes).
     */
    public int bytes() {
        return bytes;
    }

    /** For GGUF block formats: values per block (0 for the other types). */
    public int blockValues() {
        return switch (this) {
            case Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 -> 32;
            case Q4_K, Q5_K, Q6_K, Q2_K, Q3_K -> 256;
            default -> 0;
        };
    }

    /** For GGUF block formats: bytes per block. */
    public int blockBytes() {
        return switch (this) {
            case Q8_0 -> 34;
            case Q4_0 -> 18;
            case Q4_1 -> 20;
            case Q4_K -> 144;
            case Q5_K -> 176;
            case Q6_K -> 210;
            case Q5_0 -> 22;
            case Q5_1 -> 24;
            case Q2_K -> 84;
            case Q3_K -> 110;
            default -> 0;
        };
    }

    /** True for GGUF's block-quantised formats. */
    public boolean isBlockQuantized() {
        return blockValues() > 0;
    }

    /** True for the floating-point types. */
    public boolean isFloat() {
        return this != I32;
    }
}
