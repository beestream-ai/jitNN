package org.beehive.jitnn.engine;

import java.util.ArrayList;
import java.util.List;

/** The operators recorded on this thread since the last sync point, in program order. */
final class Trace {

    private static final ThreadLocal<List<Node>> PENDING = ThreadLocal.withInitial(ArrayList::new);

    private Trace() {
    }

    static void record(Node node) {
        PENDING.get().add(node);
    }

    static List<Node> drain() {
        List<Node> nodes = PENDING.get();
        PENDING.set(new ArrayList<>());
        return nodes;
    }

    /** Takes {@code node} out of this thread's pending operators if it is the last one recorded; true when it did. */
    static boolean removeLast(Node node) {
        List<Node> nodes = PENDING.get();
        if (!nodes.isEmpty() && nodes.get(nodes.size() - 1) == node) {
            nodes.remove(nodes.size() - 1);
            return true;
        }
        return false;
    }

    static List<Node> pending() {
        return PENDING.get();
    }

    static int size() {
        return PENDING.get().size();
    }
}
