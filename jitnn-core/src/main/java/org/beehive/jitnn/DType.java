package org.beehive.jitnn;

/**
 * Element type of a tensor: FP32 data, INT32 labels and indices, and the half-precision storage
 * types for mixed precision, IEEE FP16 and bfloat16 (the top 16 bits of an FP32).
 */
public enum DType {
    F32(4),
    I32(4),
    F16(2),
    BF16(2);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    public int bytes() {
        return bytes;
    }

    /** True for the floating-point types. */
    public boolean isFloat() {
        return this != I32;
    }
}
