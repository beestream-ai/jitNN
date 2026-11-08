package ai.beestream.jitnn.io;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small JSON reader for the files jitNN loads (safetensors headers, Hugging Face configs and
 * tokenizers): objects become {@code LinkedHashMap}s in file order, arrays {@code List}s, integers
 * {@code Long}, other numbers {@code Double}.
 */
public final class Json {

    private final String s;
    private int i;

    private Json(String s) {
        this.s = s;
    }

    /** Parses one JSON value. */
    public static Object parse(String text) throws IOException {
        return new Json(text).value0();
    }

    /** Parses a UTF-8 JSON file. */
    public static Object parse(java.nio.file.Path file) throws IOException {
        try {
            return parse(java.nio.file.Files.readString(file));
        } catch (IOException e) {
            throw new IOException(file + ": " + e.getMessage(), e);
        }
    }

    private Object value0() throws IOException {
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
        return new IOException("JSON: " + what + " at " + i);
    }
}
