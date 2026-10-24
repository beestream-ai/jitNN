package org.beehive.jitnn.io;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.beehive.jitnn.DType;

/**
 * Reads and writes the safetensors format (Hugging Face): an 8-byte little-endian header length,
 * a JSON header that maps each tensor name to its dtype, shape and byte range, optional
 * {@code __metadata__} string pairs, then the raw little-endian data. jitNN's checkpoints use it, so
 * they open in Python with {@code safetensors.torch.load_file}.
 */
public final class SafeTensors {

    private static final int CHUNK = 1 << 30;

    private SafeTensors() {
    }

    /** One tensor of a file: its type, shape, and where its bytes are. */
    public record Entry(DType dtype, long[] shape, long offset, long bytes) {
        public long numel() {
            long n = 1;
            for (long d : shape) {
                n *= d;
            }
            return n;
        }
    }

    /** A file's header: the tensors, in file order, and the metadata. Data offsets are absolute. */
    public record Header(Map<String, Entry> entries, Map<String, String> metadata) {
    }

    /** The data to write for one tensor: its type, shape, and a segment of exactly its bytes. */
    public record Source(DType dtype, long[] shape, MemorySegment data) {
    }

    /** Writes {@code tensors}, in iteration order, and {@code metadata} to {@code file}. */
    public static void write(Path file, Map<String, Source> tensors, Map<String, String> metadata) throws IOException {
        StringBuilder json = new StringBuilder("{");
        if (!metadata.isEmpty()) {
            json.append("\"__metadata__\":{");
            boolean first = true;
            for (Map.Entry<String, String> m : metadata.entrySet()) {
                json.append(first ? "" : ",").append(quote(m.getKey())).append(':').append(quote(m.getValue()));
                first = false;
            }
            json.append('}');
        }
        long offset = 0;
        for (Map.Entry<String, Source> e : tensors.entrySet()) {
            Source s = e.getValue();
            json.append(json.length() > 1 ? "," : "").append(quote(e.getKey())).append(":{\"dtype\":\"").append(name(s.dtype())).append("\",\"shape\":[");
            for (int i = 0; i < s.shape().length; i++) {
                json.append(i > 0 ? "," : "").append(s.shape()[i]);
            }
            long end = offset + s.data().byteSize();
            json.append("],\"data_offsets\":[").append(offset).append(',').append(end).append("]}");
            offset = end;
        }
        json.append('}');
        byte[] header = json.toString().getBytes(StandardCharsets.UTF_8);
        // the data starts 8-byte aligned, the header padded with spaces as the reference writer does
        int pad = (8 - header.length % 8) % 8;
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            ByteBuffer head = ByteBuffer.allocate(8 + header.length + pad).order(ByteOrder.LITTLE_ENDIAN);
            head.putLong(header.length + pad).put(header);
            while (head.hasRemaining()) {
                head.put((byte) ' ');
            }
            writeFully(ch, head.flip());
            for (Source s : tensors.values()) {
                MemorySegment data = s.data();
                for (long at = 0; at < data.byteSize(); at += CHUNK) {
                    writeFully(ch, data.asSlice(at, Math.min(CHUNK, data.byteSize() - at)).asByteBuffer());
                }
            }
        }
    }

    /** Reads the header of {@code file}. */
    public static Header readHeader(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer len = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            readFully(ch, len, 0);
            long n = len.flip().getLong();
            if (n < 2 || n > Math.min(100_000_000L, ch.size() - 8)) {
                throw new IOException(file + ": not a safetensors file (header length " + n + ")");
            }
            ByteBuffer json = ByteBuffer.allocate((int) n);
            readFully(ch, json, 8);
            Object root = new Json(new String(json.array(), StandardCharsets.UTF_8)).parse();
            if (!(root instanceof Map<?, ?> map)) {
                throw new IOException(file + ": header is not a JSON object");
            }
            Map<String, Entry> entries = new LinkedHashMap<>();
            Map<String, String> metadata = new LinkedHashMap<>();
            List<Map.Entry<String, Entry>> sorted = new ArrayList<>();
            for (Map.Entry<?, ?> e : map.entrySet()) {
                String key = (String) e.getKey();
                Map<?, ?> v = (Map<?, ?>) e.getValue();
                if (key.equals("__metadata__")) {
                    v.forEach((k, s) -> metadata.put((String) k, String.valueOf(s)));
                    continue;
                }
                List<?> shape = (List<?>) v.get("shape");
                List<?> range = (List<?>) v.get("data_offsets");
                long[] dims = new long[shape.size()];
                for (int i = 0; i < dims.length; i++) {
                    dims[i] = ((Number) shape.get(i)).longValue();
                }
                long begin = ((Number) range.get(0)).longValue();
                long end = ((Number) range.get(1)).longValue();
                if (end < begin || 8 + n + end > ch.size()) {
                    throw new IOException(file + ": tensor " + key + " lies outside the file");
                }
                sorted.add(Map.entry(key, new Entry(dtype((String) v.get("dtype"), key), dims, 8 + n + begin, end - begin)));
            }
            sorted.sort((a, b) -> Long.compare(a.getValue().offset(), b.getValue().offset()));
            sorted.forEach(e -> entries.put(e.getKey(), e.getValue()));
            return new Header(entries, metadata);
        }
    }

    /** Reads {@code entry}'s bytes from {@code file} into {@code target}, which must be at least that long. */
    public static void read(Path file, Entry entry, MemorySegment target) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            for (long at = 0; at < entry.bytes(); at += CHUNK) {
                readFully(ch, target.asSlice(at, Math.min(CHUNK, entry.bytes() - at)).asByteBuffer(), entry.offset() + at);
            }
        }
    }

    /** The safetensors name of {@code dtype}; GGUF block formats have none. */
    public static String name(DType dtype) {
        return switch (dtype) {
            case F32 -> "F32";
            case F16 -> "F16";
            case BF16 -> "BF16";
            case I32 -> "I32";
            case FP8_E4M3 -> "F8_E4M3";
            case FP8_E5M2 -> "F8_E5M2";
            default -> throw new IllegalArgumentException(dtype + " has no safetensors type");
        };
    }

    private static DType dtype(String name, String tensor) throws IOException {
        return switch (name) {
            case "F32" -> DType.F32;
            case "F16" -> DType.F16;
            case "BF16" -> DType.BF16;
            case "I32" -> DType.I32;
            case "F8_E4M3" -> DType.FP8_E4M3;
            case "F8_E5M2" -> DType.FP8_E5M2;
            default -> throw new IOException("tensor " + tensor + ": dtype " + name + " is not supported");
        };
    }

    private static void writeFully(FileChannel ch, ByteBuffer b) throws IOException {
        while (b.hasRemaining()) {
            ch.write(b);
        }
    }

    private static void readFully(FileChannel ch, ByteBuffer b, long position) throws IOException {
        while (b.hasRemaining()) {
            int n = ch.read(b, position);
            if (n < 0) {
                throw new IOException("unexpected end of file");
            }
            position += n;
        }
    }

    private static String quote(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) {
                        b.append(String.format("\\u%04x", (int) c));
                    } else {
                        b.append(c);
                    }
                }
            }
        }
        return b.append('"').toString();
    }

    /** The JSON subset safetensors headers use: objects, arrays, strings, numbers, true, false, null. */
    private static final class Json {
        private final String s;
        private int i;

        Json(String s) {
            this.s = s;
        }

        Object parse() throws IOException {
            Object v = value();
            space();
            if (i != s.length()) {
                throw error("trailing characters");
            }
            return v;
        }

        private Object value() throws IOException {
            space();
            if (i >= s.length()) {
                throw error("unexpected end");
            }
            char c = s.charAt(i);
            if (c == '{') {
                i++;
                Map<String, Object> m = new LinkedHashMap<>();
                space();
                if (peek('}')) {
                    return m;
                }
                do {
                    space();
                    String k = string();
                    space();
                    expect(':');
                    m.put(k, value());
                    space();
                } while (peek(','));
                expect('}');
                return m;
            }
            if (c == '[') {
                i++;
                List<Object> l = new ArrayList<>();
                space();
                if (peek(']')) {
                    return l;
                }
                do {
                    l.add(value());
                    space();
                } while (peek(','));
                expect(']');
                return l;
            }
            if (c == '"') {
                return string();
            }
            for (String word : new String[] { "true", "false", "null" }) {
                if (s.startsWith(word, i)) {
                    i += word.length();
                    return word.equals("null") ? null : Boolean.valueOf(word);
                }
            }
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) {
                i++;
            }
            if (start == i) {
                throw error("unexpected '" + c + "'");
            }
            String num = s.substring(start, i);
            return num.matches("-?\\d+") ? (Object) Long.valueOf(num) : (Object) Double.valueOf(num);
        }

        private String string() throws IOException {
            expect('"');
            StringBuilder b = new StringBuilder();
            while (i < s.length() && s.charAt(i) != '"') {
                char c = s.charAt(i++);
                if (c == '\\') {
                    char e = s.charAt(i++);
                    switch (e) {
                        case 'n' -> b.append('\n');
                        case 'r' -> b.append('\r');
                        case 't' -> b.append('\t');
                        case 'b' -> b.append('\b');
                        case 'f' -> b.append('\f');
                        case 'u' -> {
                            b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                            i += 4;
                        }
                        default -> b.append(e);
                    }
                } else {
                    b.append(c);
                }
            }
            expect('"');
            return b.toString();
        }

        private void space() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
                i++;
            }
        }

        private boolean peek(char c) {
            if (i < s.length() && s.charAt(i) == c) {
                i++;
                return true;
            }
            return false;
        }

        private void expect(char c) throws IOException {
            if (!peek(c)) {
                throw error("expected '" + c + "'");
            }
        }

        private IOException error(String what) {
            return new IOException("safetensors header: " + what + " at " + i);
        }
    }
}
