package org.beehive.jitnn;

/** Element type of a tensor. Wave 1 supports FP32 data and INT32 labels/indices. */
public enum DType {
    F32(4),
    I32(4);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    public int bytes() {
        return bytes;
    }
}
