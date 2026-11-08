package ai.beestream.jitnn.nn.text;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.io.Json;

/**
 * A Hugging Face tokenizer read from its {@code tokenizer.json}, for the two families jitNN's models
 * use: WordPiece with the BERT normaliser and pre-tokeniser (BERT and its relatives), and
 * byte-level BPE (GPT-2, RoBERTa). Added tokens such as {@code [MASK]} or {@code <|endoftext|>} are
 * matched in the raw text first; the post-processor (template, BERT or RoBERTa) adds the special
 * tokens and type ids. Other tokenizer types are rejected when loading.
 *
 * <pre>{@code
 * Tokenizer tok = Tokenizer.load(Path.of("bert-base-uncased"));
 * Tokenizer.Batch batch = tok.encodeBatch(List.of("Hello world!", "A longer sentence."), 128);
 * Tensor hidden = bert.encode(batch.ids(), batch.typeIds(), batch.additiveMask());
 * }</pre>
 */
public final class Tokenizer {

    private enum Model {
        WORDPIECE,
        BPE
    }

    /** An added token; lstrip / rstrip: it takes the whitespace on its left / right with it. */
    private record Added(String content, int id, boolean special, boolean lstrip, boolean rstrip) {
    }

    /** One item of a post-processing template: a special token, or sequence 0 (A) or 1 (B). */
    private record Item(String special, int sequence, int typeId) {
    }

    /** The ids, type ids and tokens of one text (or pair), special tokens included. */
    public record Encoding(int[] ids, int[] typeIds, String[] tokens) {
    }

    /**
     * Encodings padded to the longest one: {@code [batch, length]} ids and type ids, and an
     * attention mask of 1 for tokens and 0 for padding.
     */
    public record Batch(int[] idArray, int[] typeIdArray, int[] attention, int batch, int length) {

        /** The token ids as an INT32 tensor {@code [batch, length]}. */
        public Tensor ids() {
            return Tensor.of(idArray, batch, length);
        }

        /** The token type (segment) ids as an INT32 tensor {@code [batch, length]}. */
        public Tensor typeIds() {
            return Tensor.of(typeIdArray, batch, length);
        }

        /** An additive attention mask [batch, 1, 1, length]: 0 for tokens, -10000 for padding. */
        public Tensor additiveMask() {
            float[] m = new float[attention.length];
            for (int i = 0; i < m.length; i++) {
                m[i] = attention[i] == 0 ? -10000.0f : 0.0f;
            }
            return Tensor.of(m, batch, 1, 1, length);
        }
    }

    private static final Pattern GPT2_SPLIT = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+", Pattern.UNICODE_CHARACTER_CLASS);
    private static final char[] BYTE_TO_CHAR = new char[256];
    private static final Map<Character, Integer> CHAR_TO_BYTE = new HashMap<>();

    static {
        // GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to 256 + n
        int n = 0;
        for (int b = 0; b < 256; b++) {
            boolean printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            BYTE_TO_CHAR[b] = printable ? (char) b : (char) (256 + n++);
            CHAR_TO_BYTE.put(BYTE_TO_CHAR[b], b);
        }
    }

    private final Model model;
    private final Map<String, Integer> vocab = new HashMap<>();
    private final String[] tokens;
    private final List<Added> added = new ArrayList<>();
    private final Map<Integer, Added> addedById = new HashMap<>();
    private Integer unk;

    // WordPiece
    private String subwordPrefix = "##";
    private int maxWordChars = 100;
    private boolean bertNormalizer;
    private boolean cleanText;
    private boolean chineseChars;
    private boolean stripAccents;
    private boolean lowercase;

    // BPE
    private final Map<String, Integer> mergeRanks = new HashMap<>();
    private boolean byteLevel;
    private boolean addPrefixSpace;
    private boolean useRegex = true;
    private boolean ignoreMerges;
    /** Splits the pre-tokenizer applies before the byte-level step, in order. */
    private final List<java.util.function.Function<String, List<String>>> splits = new ArrayList<>();
    /** A Unicode normal form applied before the other normalisers, or null. */
    private Normalizer.Form unicodeForm;
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private List<Item> single = List.of(new Item(null, 0, 0));
    private List<Item> pair = List.of(new Item(null, 0, 0), new Item(null, 1, 1));
    private final Map<String, int[]> specialIds = new HashMap<>();
    private Integer padId;

    private Tokenizer(Map<?, ?> root) {
        Map<?, ?> m = map(root.get("model"), "model");
        // older files leave the type out; the reference implementation tells the models apart by their fields
        String type = m.get("type") instanceof String t ? t
                : m.containsKey("merges") ? "BPE" : m.containsKey("max_input_chars_per_word") ? "WordPiece" : m.get("vocab") instanceof List<?> ? "Unigram" : "WordLevel";
        this.model = switch (type) {
            case "WordPiece" -> Model.WORDPIECE;
            case "BPE" -> Model.BPE;
            default -> throw new IllegalArgumentException("tokenizer model " + type + " is not supported (WordPiece and BPE are)");
        };
        int max = -1;
        for (Map.Entry<?, ?> e : map(m.get("vocab"), "model.vocab").entrySet()) {
            int id = ((Number) e.getValue()).intValue();
            vocab.put((String) e.getKey(), id);
            max = Math.max(max, id);
        }
        List<?> addedTokens = root.get("added_tokens") instanceof List<?> l ? l : List.of();
        for (Object o : addedTokens) {
            Map<?, ?> a = (Map<?, ?>) o;
            Added t = new Added((String) a.get("content"), ((Number) a.get("id")).intValue(), Boolean.TRUE.equals(a.get("special")), Boolean.TRUE.equals(a.get("lstrip")),
                    Boolean.TRUE.equals(a.get("rstrip")));
            added.add(t);
            addedById.put(t.id(), t);
            max = Math.max(max, t.id());
        }
        // longest first, so "[MASK]" wins over a shorter token that is its prefix
        added.sort((a, b) -> Integer.compare(b.content().length(), a.content().length()));
        this.tokens = new String[max + 1];
        vocab.forEach((k, v) -> tokens[v] = k);
        for (Added a : added) {
            tokens[a.id()] = a.content();
        }
        if (m.get("unk_token") instanceof String u) {
            unk = id(u);
        }
        if (model == Model.WORDPIECE) {
            if (m.get("continuing_subword_prefix") instanceof String p) {
                subwordPrefix = p;
            }
            if (m.get("max_input_chars_per_word") instanceof Number n) {
                maxWordChars = n.intValue();
            }
        } else {
            for (Object o : (List<?>) m.get("merges")) {
                String key = o instanceof List<?> p ? p.get(0) + " " + p.get(1) : (String) o;
                mergeRanks.putIfAbsent(key, mergeRanks.size());
            }
            ignoreMerges = Boolean.TRUE.equals(m.get("ignore_merges"));
            if (m.get("dropout") instanceof Number d && d.doubleValue() > 0) {
                throw new IllegalArgumentException("BPE dropout is not supported");
            }
            if (Boolean.TRUE.equals(m.get("byte_fallback"))) {
                throw new IllegalArgumentException("BPE byte fallback is not supported");
            }
        }
        normalizer(root.get("normalizer"));
        preTokenizer(root.get("pre_tokenizer"));
        postProcessor(root.get("post_processor"));
        if (root.get("padding") instanceof Map<?, ?> p && p.get("pad_id") instanceof Number id) {
            padId = id.intValue();
        }
    }

    /** Loads {@code tokenizer.json}, or the one inside directory {@code path}. */
    public static Tokenizer load(Path path) {
        Path file = Files.isDirectory(path) ? path.resolve("tokenizer.json") : path;
        try {
            return new Tokenizer(map(Json.parse(file), file.toString()));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private void normalizer(Object o) {
        if (o == null) {
            return;
        }
        Map<?, ?> n = map(o, "normalizer");
        switch (String.valueOf(n.get("type"))) {
            case "BertNormalizer" -> {
                bertNormalizer = true;
                cleanText = !Boolean.FALSE.equals(n.get("clean_text"));
                chineseChars = !Boolean.FALSE.equals(n.get("handle_chinese_chars"));
                lowercase = !Boolean.FALSE.equals(n.get("lowercase"));
                // null strip_accents follows lowercase, as in the reference implementation
                stripAccents = n.get("strip_accents") == null ? lowercase : Boolean.TRUE.equals(n.get("strip_accents"));
            }
            case "Sequence" -> {
                for (Object each : (List<?>) n.get("normalizers")) {
                    normalizer(each);
                }
            }
            case "Lowercase" -> lowercase = true;
            case "NFC" -> unicodeForm = Normalizer.Form.NFC;
            case "NFKC" -> unicodeForm = Normalizer.Form.NFKC;
            case "StripAccents" -> stripAccents = true;
            case "NFD" -> {
                // StripAccents, the only reason tokenizers use it here, decomposes itself
            }
            default -> throw new IllegalArgumentException("normalizer " + n.get("type") + " is not supported");
        }
    }

    private void preTokenizer(Object o) {
        if (o == null) {
            return;
        }
        preTokenizerStep(o);
        if (model == Model.BPE && !byteLevel) {
            throw new IllegalArgumentException("BPE without the ByteLevel pre-tokenizer is not supported");
        }
    }

    private void preTokenizerStep(Object o) {
        Map<?, ?> p = map(o, "pre_tokenizer");
        switch (String.valueOf(p.get("type"))) {
            case "Sequence" -> {
                for (Object each : (List<?>) p.get("pretokenizers")) {
                    preTokenizerStep(each);
                }
            }
            case "Digits" -> {
                // each digit, or each run of digits, becomes a piece of its own
                Pattern digits = Pattern.compile(Boolean.TRUE.equals(p.get("individual_digits")) ? "\\p{N}" : "\\p{N}+", Pattern.UNICODE_CHARACTER_CLASS);
                splits.add(t -> isolate(digits, t));
            }
            case "Split" -> {
                Map<?, ?> pattern = map(p.get("pattern"), "Split pattern");
                if (!"Isolated".equals(p.get("behavior")) || Boolean.TRUE.equals(p.get("invert"))) {
                    throw new IllegalArgumentException("Split pre-tokenizer: only the Isolated behaviour without invert is supported");
                }
                Pattern re = pattern.get("Regex") instanceof String r ? Pattern.compile(r, Pattern.UNICODE_CHARACTER_CLASS)
                        : Pattern.compile(Pattern.quote((String) pattern.get("String")));
                splits.add(t -> isolate(re, t));
            }
            case "BertPreTokenizer" -> {
                if (model != Model.WORDPIECE) {
                    throw new IllegalArgumentException("BertPreTokenizer with a BPE model is not supported");
                }
            }
            case "ByteLevel" -> {
                byteLevel = true;
                addPrefixSpace = Boolean.TRUE.equals(p.get("add_prefix_space"));
                useRegex = !Boolean.FALSE.equals(p.get("use_regex"));
            }
            default -> throw new IllegalArgumentException("pre-tokenizer " + p.get("type") + " is not supported");
        }
    }

    /** The matches of {@code re} in t and the text between them, in order, as separate pieces (empty pieces dropped). */
    private static List<String> isolate(Pattern re, String t) {
        List<String> out = new ArrayList<>();
        Matcher m = re.matcher(t);
        int last = 0;
        while (m.find()) {
            if (m.start() > last) {
                out.add(t.substring(last, m.start()));
            }
            if (m.end() > m.start()) {
                out.add(m.group());
            }
            last = m.end();
        }
        if (last < t.length()) {
            out.add(t.substring(last));
        }
        return out;
    }

    private void postProcessor(Object o) {
        if (o == null) {
            return;
        }
        Map<?, ?> p = map(o, "post_processor");
        switch (String.valueOf(p.get("type"))) {
            case "TemplateProcessing" -> {
                single = template((List<?>) p.get("single"));
                pair = template((List<?>) p.get("pair"));
                for (Map.Entry<?, ?> e : map(p.get("special_tokens"), "special_tokens").entrySet()) {
                    List<?> ids = (List<?>) ((Map<?, ?>) e.getValue()).get("ids");
                    specialIds.put((String) e.getKey(), ids.stream().mapToInt(x -> ((Number) x).intValue()).toArray());
                }
            }
            case "BertProcessing", "RobertaProcessing" -> {
                List<?> cls = (List<?>) p.get("cls");
                List<?> sep = (List<?>) p.get("sep");
                String c = (String) cls.get(0);
                String s = (String) sep.get(0);
                specialIds.put(c, new int[] { ((Number) cls.get(1)).intValue() });
                specialIds.put(s, new int[] { ((Number) sep.get(1)).intValue() });
                single = List.of(new Item(c, -1, 0), new Item(null, 0, 0), new Item(s, -1, 0));
                boolean bert = p.get("type").equals("BertProcessing");
                pair = bert ? List.of(new Item(c, -1, 0), new Item(null, 0, 0), new Item(s, -1, 0), new Item(null, 1, 1), new Item(s, -1, 1))
                        : List.of(new Item(c, -1, 0), new Item(null, 0, 0), new Item(s, -1, 0), new Item(s, -1, 0), new Item(null, 1, 0), new Item(s, -1, 0));
            }
            case "ByteLevel" -> {
                // only adjusts offsets
            }
            case "Sequence" -> {
                for (Object each : (List<?>) p.get("processors")) {
                    postProcessor(each);
                }
            }
            default -> throw new IllegalArgumentException("post-processor " + p.get("type") + " is not supported");
        }
    }

    private static List<Item> template(List<?> items) {
        List<Item> out = new ArrayList<>();
        for (Object o : items) {
            Map<?, ?> i = (Map<?, ?>) o;
            if (i.get("SpecialToken") instanceof Map<?, ?> s) {
                out.add(new Item((String) s.get("id"), -1, ((Number) s.get("type_id")).intValue()));
            } else {
                Map<?, ?> s = (Map<?, ?>) i.get("Sequence");
                out.add(new Item(null, s.get("id").equals("A") ? 0 : 1, ((Number) s.get("type_id")).intValue()));
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- encoding

    /** Encodes one text, with the post-processor's special tokens. */
    public Encoding encode(String text) {
        return encode(text, null, Integer.MAX_VALUE);
    }

    /** Encodes a pair of texts (sentence A, sentence B). */
    public Encoding encode(String a, String b) {
        return encode(a, b, Integer.MAX_VALUE);
    }

    /**
     * Encodes {@code a} (and {@code b} when not null) into at most {@code maxLength} ids, special
     * tokens included, cutting the longer sequence first as Hugging Face's longest-first truncation.
     */
    public Encoding encode(String a, String b, int maxLength) {
        List<Item> template = b == null ? single : pair;
        int[][] seq = { tokenize(a), b == null ? new int[0] : tokenize(b) };
        int specials = 0;
        for (Item i : template) {
            if (i.special() != null) {
                specials += specialIds(i.special()).length;
            }
        }
        int budget = Math.max(maxLength - specials, 0);
        while (seq[0].length + seq[1].length > budget) {
            int longer = seq[0].length >= seq[1].length ? 0 : 1;
            seq[longer] = Arrays.copyOf(seq[longer], seq[longer].length - 1);
        }
        int n = 0;
        for (Item i : template) {
            n += i.special() != null ? specialIds(i.special()).length : seq[i.sequence()].length;
        }
        int[] ids = new int[n];
        int[] types = new int[n];
        int at = 0;
        for (Item i : template) {
            int[] part = i.special() != null ? specialIds(i.special()) : seq[i.sequence()];
            System.arraycopy(part, 0, ids, at, part.length);
            Arrays.fill(types, at, at + part.length, i.typeId());
            at += part.length;
        }
        String[] toks = new String[n];
        for (int i = 0; i < n; i++) {
            toks[i] = token(ids[i]);
        }
        return new Encoding(ids, types, toks);
    }

    /**
     * Encodes {@code texts}, each cut to {@code maxLength}, padded to the longest with
     * {@link #padId()}.
     */
    public Batch encodeBatch(List<String> texts, int maxLength) {
        List<Encoding> encs = new ArrayList<>();
        int len = 1;
        for (String t : texts) {
            Encoding e = encode(t, null, maxLength);
            encs.add(e);
            len = Math.max(len, e.ids().length);
        }
        int[] ids = new int[texts.size() * len];
        int[] types = new int[ids.length];
        int[] attention = new int[ids.length];
        Arrays.fill(ids, padId());
        for (int r = 0; r < encs.size(); r++) {
            Encoding e = encs.get(r);
            System.arraycopy(e.ids(), 0, ids, r * len, e.ids().length);
            System.arraycopy(e.typeIds(), 0, types, r * len, e.ids().length);
            Arrays.fill(attention, r * len, r * len + e.ids().length, 1);
        }
        return new Batch(ids, types, attention, texts.size(), len);
    }

    /** The ids of {@code text} without special tokens. */
    public int[] tokenize(String text) {
        List<Integer> out = new ArrayList<>();
        int start = 0;
        int i = 0;
        while (i < text.length()) {
            Added hit = null;
            for (Added a : added) {
                if (text.startsWith(a.content(), i)) {
                    hit = a;
                    break;
                }
            }
            if (hit == null) {
                i++;
                continue;
            }
            int end = i;
            if (hit.lstrip()) {
                while (end > start && Character.isWhitespace(text.charAt(end - 1))) {
                    end--;
                }
            }
            model(text.substring(start, end), out);
            out.add(hit.id());
            i += hit.content().length();
            if (hit.rstrip()) {
                while (i < text.length() && Character.isWhitespace(text.charAt(i))) {
                    i++;
                }
            }
            start = i;
        }
        model(text.substring(start), out);
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    private void model(String text, List<Integer> out) {
        if (text.isEmpty()) {
            return;
        }
        if (model == Model.WORDPIECE) {
            for (String word : bertWords(normalize(text))) {
                wordPiece(word, out);
            }
        } else {
            String t = normalize(text);
            if (addPrefixSpace && !t.isEmpty() && !Character.isWhitespace(t.charAt(0))) {
                t = " " + t;
            }
            List<String> pieces = new ArrayList<>(List.of(t));
            for (var split : splits) {
                List<String> next = new ArrayList<>();
                for (String piece : pieces) {
                    next.addAll(split.apply(piece));
                }
                pieces = next;
            }
            if (useRegex) {
                List<String> next = new ArrayList<>();
                for (String piece : pieces) {
                    Matcher m = GPT2_SPLIT.matcher(piece);
                    while (m.find()) {
                        next.add(m.group());
                    }
                }
                pieces = next;
            }
            for (String piece : pieces) {
                for (int id : bpe(byteLevel(piece))) {
                    out.add(id);
                }
            }
        }
    }

    private String normalize(String text) {
        String t = unicodeForm != null ? Normalizer.normalize(text, unicodeForm) : text;
        if (bertNormalizer) {
            StringBuilder b = new StringBuilder(t.length());
            t.codePoints().forEach(c -> {
                if (cleanText && (c == 0 || c == 0xFFFD || isControl(c))) {
                    return;
                }
                if (cleanText && isWhitespace(c)) {
                    b.append(' ');
                } else if (chineseChars && isChinese(c)) {
                    b.append(' ').appendCodePoint(c).append(' ');
                } else {
                    b.appendCodePoint(c);
                }
            });
            t = b.toString();
        }
        if (stripAccents) {
            StringBuilder b = new StringBuilder();
            Normalizer.normalize(t, Normalizer.Form.NFD).codePoints().filter(c -> Character.getType(c) != Character.NON_SPACING_MARK).forEach(b::appendCodePoint);
            t = b.toString();
        }
        if (lowercase) {
            t = t.toLowerCase(Locale.ROOT);
        }
        return t;
    }

    /** BERT's pre-tokeniser: split on whitespace, and every punctuation character on its own. */
    private static List<String> bertWords(String text) {
        List<String> words = new ArrayList<>();
        StringBuilder w = new StringBuilder();
        text.codePoints().forEach(c -> {
            if (isWhitespace(c)) {
                flush(w, words);
            } else if (isPunctuation(c)) {
                flush(w, words);
                words.add(new String(Character.toChars(c)));
            } else {
                w.appendCodePoint(c);
            }
        });
        flush(w, words);
        return words;
    }

    private static void flush(StringBuilder w, List<String> words) {
        if (w.length() > 0) {
            words.add(w.toString());
            w.setLength(0);
        }
    }

    private void wordPiece(String word, List<Integer> out) {
        int[] cps = word.codePoints().toArray();
        if (cps.length > maxWordChars) {
            out.add(requireUnk());
            return;
        }
        List<Integer> pieces = new ArrayList<>();
        int start = 0;
        while (start < cps.length) {
            int end = cps.length;
            Integer found = null;
            while (start < end) {
                String sub = (start > 0 ? subwordPrefix : "") + new String(cps, start, end - start);
                found = vocab.get(sub);
                if (found != null) {
                    break;
                }
                end--;
            }
            if (found == null) {
                out.add(requireUnk());
                return;
            }
            pieces.add(found);
            start = end;
        }
        out.addAll(pieces);
    }

    private static String byteLevel(String piece) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        char[] c = new char[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            c[i] = BYTE_TO_CHAR[bytes[i] & 0xFF];
        }
        return new String(c);
    }

    private int[] bpe(String word) {
        int[] hit = cache.get(word);
        if (hit != null) {
            return hit;
        }
        List<String> sym = new ArrayList<>();
        if (ignoreMerges && vocab.containsKey(word)) {
            sym.add(word);
        } else {
            word.codePoints().forEach(c -> sym.add(new String(Character.toChars(c))));
            while (sym.size() > 1) {
                int best = Integer.MAX_VALUE;
                int at = -1;
                for (int i = 0; i + 1 < sym.size(); i++) {
                    Integer r = mergeRanks.get(sym.get(i) + " " + sym.get(i + 1));
                    if (r != null && r < best) {
                        best = r;
                        at = i;
                    }
                }
                if (at < 0) {
                    break;
                }
                String left = sym.get(at);
                String right = sym.get(at + 1);
                List<String> next = new ArrayList<>(sym.size());
                for (int i = 0; i < sym.size(); i++) {
                    if (i + 1 < sym.size() && sym.get(i).equals(left) && sym.get(i + 1).equals(right)) {
                        next.add(left + right);
                        i++;
                    } else {
                        next.add(sym.get(i));
                    }
                }
                sym.clear();
                sym.addAll(next);
            }
        }
        int[] ids = new int[sym.size()];
        for (int i = 0; i < ids.length; i++) {
            Integer id = vocab.get(sym.get(i));
            ids[i] = id != null ? id : requireUnk();
        }
        if (cache.size() < 100_000) {
            cache.put(word, ids);
        }
        return ids;
    }

    // ---------------------------------------------------------------- decoding

    /** The text of {@code ids}; with {@code skipSpecial}, special added tokens are left out. */
    public String decode(int[] ids, boolean skipSpecial) {
        List<String> parts = new ArrayList<>();
        for (int id : ids) {
            Added a = addedById.get(id);
            if (skipSpecial && a != null && a.special()) {
                continue;
            }
            parts.add(token(id));
        }
        if (model == Model.BPE) {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            for (String p : parts) {
                if (addedById.containsKey(id(p))) {
                    bytes.writeBytes(p.getBytes(StandardCharsets.UTF_8));
                    continue;
                }
                for (char c : p.toCharArray()) {
                    Integer b = CHAR_TO_BYTE.get(c);
                    if (b != null) {
                        bytes.write(b);
                    }
                }
            }
            return bytes.toString(StandardCharsets.UTF_8);
        }
        // the WordPiece decoder: subwords join their word, other tokens get a space, and each token
        // is cleaned up on its own (so " ." loses its space but " ' t" keeps them), as the reference does
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) {
            String p = parts.get(i);
            p = p.startsWith(subwordPrefix) ? p.substring(subwordPrefix.length()) : i > 0 ? " " + p : p;
            text.append(p.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'").replace(" n't", "n't").replace(" 'm", "'m")
                    .replace(" do not", " don't").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re"));
        }
        return text.toString();
    }

    // ---------------------------------------------------------------- vocabulary

    /** The number of tokens in the vocabulary. */
    public int vocabSize() {
        return tokens.length;
    }

    /** The id of {@code token}, or -1. */
    public int id(String token) {
        Integer v = vocab.get(token);
        if (v != null) {
            return v;
        }
        for (Added a : added) {
            if (a.content().equals(token)) {
                return a.id();
            }
        }
        return -1;
    }

    /** The token with id {@code id}, or null. */
    public String token(int id) {
        return id >= 0 && id < tokens.length ? tokens[id] : null;
    }

    /**
     * The id used for padding: the file's padding id, else {@code [PAD]} or {@code <pad>}, else
     * GPT-2's {@code <|endoftext|>}, else 0. Padded positions are masked out by {@link Batch#attention}.
     */
    public int padId() {
        if (padId != null) {
            return padId;
        }
        for (String t : new String[] { "[PAD]", "<pad>", "<|endoftext|>" }) {
            if (id(t) >= 0) {
                return id(t);
            }
        }
        return 0;
    }

    private int[] specialIds(String token) {
        int[] ids = specialIds.get(token);
        if (ids == null) {
            int id = id(token);
            if (id < 0) {
                throw new IllegalStateException("special token " + token + " is not in the vocabulary");
            }
            ids = new int[] { id };
        }
        return ids;
    }

    private int requireUnk() {
        if (unk == null || unk < 0) {
            throw new IllegalArgumentException("text has a token outside the vocabulary and the tokenizer has no unknown token");
        }
        return unk;
    }

    // ---------------------------------------------------------------- character classes (as the reference implementation)

    private static boolean isWhitespace(int c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r' || Character.isWhitespace(c) || Character.isSpaceChar(c);
    }

    private static boolean isControl(int c) {
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }
        int t = Character.getType(c);
        return t == Character.CONTROL || t == Character.FORMAT || t == Character.PRIVATE_USE || t == Character.SURROGATE || t == Character.UNASSIGNED;
    }

    private static boolean isPunctuation(int c) {
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        int t = Character.getType(c);
        return t == Character.CONNECTOR_PUNCTUATION || t == Character.DASH_PUNCTUATION || t == Character.START_PUNCTUATION || t == Character.END_PUNCTUATION
                || t == Character.INITIAL_QUOTE_PUNCTUATION || t == Character.FINAL_QUOTE_PUNCTUATION || t == Character.OTHER_PUNCTUATION;
    }

    private static boolean isChinese(int c) {
        return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81F)
                || (c >= 0x2B820 && c <= 0x2CEAF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
    }

    private static Map<?, ?> map(Object o, String what) {
        if (!(o instanceof Map<?, ?> m)) {
            throw new IllegalArgumentException(what + " is not a JSON object");
        }
        return m;
    }
}
