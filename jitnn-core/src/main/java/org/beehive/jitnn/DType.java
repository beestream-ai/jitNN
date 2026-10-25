package org.beehive.jitnn;

/**
 * Element type of a tensor: FP32 data, INT32 labels and indices, and the half-precision storage
 * types for mixed precision, IEEE FP16 and bfloat16 (the top 16 bits of an FP32), and Q8_0, GGUF's
 * 8-bit blocks (per 32 consecutive values an FP16 scale and 32 signed bytes, 34 bytes), and the
 * OCP 8-bit floats FP8 E4M3 and E5M2, one byte per value, stored without their scale (see
 * {@link org.beehive.jitnn.ops.Fp8}). Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q2_K and Q3_K are GGUF's other block
 * formats, stored as their raw blocks (see {@link org.beehive.jitnn.ops.Gguf}).
 */
public enum DType {
    F32(4),
    I32(4),
    F16(2),
    BF16(2),
    Q8_0(1),
    FP8_E4M3(1),
    FP8_E5M2(1),
    Q4_0(1),
    Q4_1(1),
    Q4_K(1),
    Q5_K(1),
    Q6_K(1),
    Q5_0(1),
    Q5_1(1),
    Q2_K(1),
    Q3_K(1);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

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
