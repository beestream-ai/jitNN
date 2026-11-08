package ai.beestream.jitnn.tf;

/** Small helpers on TensorFlow tensor references. */
final class TfGraphs {

    private TfGraphs() {
    }

    /** "node" as "node:0"; "node:k" unchanged; control inputs keep their "^". */
    static String canonical(String ref) {
        if (ref.startsWith("^") || ref.lastIndexOf(':') > 0) {
            return ref;
        }
        return ref + ":0";
    }

    /** The node a reference reads. */
    static String node(String ref) {
        int c = ref.lastIndexOf(':');
        return c > 0 ? ref.substring(0, c) : ref;
    }
}
