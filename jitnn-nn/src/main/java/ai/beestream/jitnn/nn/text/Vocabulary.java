package ai.beestream.jitnn.nn.text;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Word ids by descending frequency, the words seen at least {@code minCount} times. */
public final class Vocabulary {

    private final Map<String, Integer> ids = new HashMap<>();
    private final List<String> words = new ArrayList<>();
    private final long[] counts;

    private Vocabulary(Map<String, Long> freq, int minCount) {
        List<Map.Entry<String, Long>> kept = new ArrayList<>();
        for (var e : freq.entrySet()) {
            if (e.getValue() >= minCount) {
                kept.add(e);
            }
        }
        kept.sort((a, b) -> a.getValue().equals(b.getValue()) ? a.getKey().compareTo(b.getKey()) : Long.compare(b.getValue(), a.getValue()));
        counts = new long[kept.size()];
        for (var e : kept) {
            counts[words.size()] = e.getValue();
            ids.put(e.getKey(), words.size());
            words.add(e.getKey());
        }
    }

    /** The vocabulary of the words in {@code sentences} seen at least {@code minCount} times. */
    public static Vocabulary build(List<String[]> sentences, int minCount) {
        Map<String, Long> freq = new HashMap<>();
        for (String[] s : sentences) {
            for (String w : s) {
                freq.merge(w, 1L, Long::sum);
            }
        }
        return new Vocabulary(freq, minCount);
    }

    /** The number of words. */
    public int size() {
        return words.size();
    }

    /** The id of {@code word}, or -1 when it is not in the vocabulary. */
    public int id(String word) {
        return ids.getOrDefault(word, -1);
    }

    /** The word with id {@code id}. */
    public String word(int id) {
        return words.get(id);
    }

    /** Each word's count, indexed by id. */
    public long[] counts() {
        return counts.clone();
    }

    /** Each sentence as ids, words outside the vocabulary dropped. */
    public List<int[]> encode(List<String[]> sentences) {
        List<int[]> out = new ArrayList<>();
        for (String[] s : sentences) {
            out.add(Arrays.stream(s).mapToInt(this::id).filter(i -> i >= 0).toArray());
        }
        return out;
    }
}
