package org.beehive.jitnn.onnx;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * A reader for the protocol buffers wire format, enough to decode ONNX's messages without a
 * protobuf library: varints, fixed 32/64-bit values and length-delimited fields (strings, bytes,
 * nested messages and packed repeated numbers).
 */
final class Proto {

    static final int VARINT = 0;
    static final int FIXED64 = 1;
    static final int LENGTH = 2;
    static final int FIXED32 = 5;

    private final ByteBuffer buf;
    private final int end;
    private int tag;

    Proto(ByteBuffer buf) {
        this.buf = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        this.end = this.buf.limit();
    }

    private Proto(ByteBuffer buf, int start, int end) {
        this.buf = buf.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        this.buf.position(start);
        this.buf.limit(end);
        this.end = end;
    }

    /** Advances to the next field; false at the end of the message. */
    boolean next() {
        if (buf.position() >= end) {
            return false;
        }
        tag = (int) varint();
        return true;
    }

    int field() {
        return tag >>> 3;
    }

    int wire() {
        return tag & 7;
    }

    long varint() {
        long result = 0;
        int shift = 0;
        while (true) {
            byte b = buf.get();
            result |= (long) (b & 0x7F) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
            if (shift > 63) {
                throw new IllegalStateException("malformed varint in ONNX model");
            }
        }
    }

    long fixed64() {
        return buf.getLong();
    }

    int fixed32() {
        return buf.getInt();
    }

    float float32() {
        return buf.getFloat();
    }

    double float64() {
        return buf.getDouble();
    }

    /** The current length-delimited field as a view (no copy). */
    ByteBuffer bytes() {
        int len = (int) varint();
        ByteBuffer slice = buf.slice(buf.position(), len).order(ByteOrder.LITTLE_ENDIAN);
        buf.position(buf.position() + len);
        return slice;
    }

    String string() {
        ByteBuffer b = bytes();
        byte[] a = new byte[b.remaining()];
        b.get(a);
        return new String(a, StandardCharsets.UTF_8);
    }

    /** The current length-delimited field as a nested message. */
    Proto message() {
        int len = (int) varint();
        int start = buf.position();
        buf.position(start + len);
        return new Proto(buf, start, start + len);
    }

    /** Repeated int64 values: one varint, or a packed run. */
    void longs(LongSink sink) {
        if (wire() == LENGTH) {
            int len = (int) varint();
            int stop = buf.position() + len;
            while (buf.position() < stop) {
                sink.accept(varint());
            }
        } else {
            sink.accept(varint());
        }
    }

    /** Repeated float values: one fixed32, or a packed run. */
    void floats(FloatSink sink) {
        if (wire() == LENGTH) {
            int len = (int) varint();
            int stop = buf.position() + len;
            while (buf.position() < stop) {
                sink.accept(buf.getFloat());
            }
        } else {
            sink.accept(buf.getFloat());
        }
    }

    /** Repeated double values: one fixed64, or a packed run. */
    void doubles(DoubleSink sink) {
        if (wire() == LENGTH) {
            int len = (int) varint();
            int stop = buf.position() + len;
            while (buf.position() < stop) {
                sink.accept(buf.getDouble());
            }
        } else {
            sink.accept(buf.getDouble());
        }
    }

    void skip() {
        switch (wire()) {
            case VARINT -> varint();
            case FIXED64 -> buf.position(buf.position() + 8);
            case LENGTH -> {
                int len = (int) varint();
                buf.position(buf.position() + len);
            }
            case FIXED32 -> buf.position(buf.position() + 4);
            default -> throw new IllegalStateException("unsupported protobuf wire type " + wire() + " in ONNX model");
        }
    }

    interface LongSink {
        void accept(long v);
    }

    interface FloatSink {
        void accept(float v);
    }

    interface DoubleSink {
        void accept(double v);
    }
}
