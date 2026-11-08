package ai.beestream.jitnn.engine;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Per-GPU tuning tables: the kernel variant to use for each shape class, measured offline by
 * sweeping every variant over the reference workloads (scripts/tune.py in jitNN-benchmarking) and read when a plan
 * is compiled.
 *
 * <p>A table is {@code tuning/<backend>-<device>.json} on the class path, e.g.
 * {@code tuning/cuda-nvidia-a100-80gb-pcie.json}, or the file named by {@code -Djitnn.tuning=};
 * {@code -Djitnn.tuning=off} ignores tables. Its {@code "gemm"} object maps a shape class to a
 * variant name of {@code MatMul}: the class is {@code m<a>n<b>k<c>t<ta><tb>} with a, b, c the
 * ceiling log2 of M, N and K, so shapes of similar size share an entry. An exact shape
 * {@code <m>x<n>x<k>t<ta><tb>} overrides its class where the class's variant was measured slower
 * for it. Shapes without an entry keep the built-in choice, as do all shapes when no table
 * exists for the device.
 */
public final class Tuning {

    private static final String SETTING = System.getProperty("jitnn.tuning", "").trim();
    private static final Map<String, Map<String, String>> TABLES = new ConcurrentHashMap<>();
    private static final Pattern ENTRY = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"");

    private Tuning() {
    }

    /** The table's GEMM variant for this shape on the current device (its exact entry, else its class's), or null. */
    public static String gemm(int m, int n, int k, int ta, int tb) {
        Map<String, String> t = table("gemm");
        String exact = t.get(gemmShape(m, n, k, ta, tb));
        return exact != null ? exact : t.get(gemmClass(m, n, k, ta, tb));
    }

    /** The exact-shape key of a GEMM, for the entries that override their class. */
    public static String gemmShape(int m, int n, int k, int ta, int tb) {
        return m + "x" + n + "x" + k + "t" + ta + tb;
    }

    /** The shape class of a GEMM: ceiling log2 of M, N and K, and the transposes. */
    public static String gemmClass(int m, int n, int k, int ta, int tb) {
        return "m" + log2(m) + "n" + log2(n) + "k" + log2(k) + "t" + ta + tb;
    }

    static int log2(int x) {
        return x <= 1 ? 0 : 32 - Integer.numberOfLeadingZeros(x - 1);
    }

    /** The file name of the current device's table: backend and device name, lower case, dashes. */
    public static String fileName() {
        String id = Devices.backend().name() + "-" + Devices.name();
        return id.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-").replaceAll("(^-|-$)", "") + ".json";
    }

    private static Map<String, String> table(String section) {
        if (SETTING.equals("off")) {
            return Map.of();
        }
        String file = SETTING.isEmpty() ? fileName() : SETTING;
        return TABLES.computeIfAbsent(file + "#" + section, k -> parse(read(file), section));
    }

    private static String read(String file) {
        try {
            if (!SETTING.isEmpty()) {
                return Files.readString(Path.of(file));
            }
            try (InputStream in = Tuning.class.getResourceAsStream("/tuning/" + file)) {
                return in == null ? "" : new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
        } catch (IOException e) {
            throw new IllegalStateException("cannot read tuning table " + file, e);
        }
    }

    /** The string entries of one top-level object of the table (the format tune.py writes). */
    static Map<String, String> parse(String json, String section) {
        Map<String, String> out = new HashMap<>();
        int at = json.indexOf("\"" + section + "\"");
        if (at < 0) {
            return out;
        }
        int open = json.indexOf('{', at);
        int close = json.indexOf('}', open);
        if (open < 0 || close < 0) {
            return out;
        }
        Matcher m = ENTRY.matcher(json.substring(open + 1, close));
        while (m.find()) {
            out.put(m.group(1), m.group(2));
        }
        return out;
    }
}
