package ai.beestream.jitnn.nn.models;

/**
 * Hands out initialisation seeds in construction order (base, base + 1, ...), so a model can be
 * rebuilt with identical weights elsewhere (scripts/pytorch/models.py follows the same order).
 */
final class Seeds {

    private long next;

    Seeds(long base) {
        this.next = base;
    }

    long next() {
        return next++;
    }
}
