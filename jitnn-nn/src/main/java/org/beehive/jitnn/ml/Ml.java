package org.beehive.jitnn.ml;

import java.util.Arrays;
import java.util.TreeSet;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.engine.Engine;

/** Shared pieces of the estimators: label encoding, distances, and running a step. */
final class Ml {

    private Ml() {
    }

    /** Squared Euclidean distances [n, m] between the rows of a [n, d] and b [m, d]. */
    static Tensor sqDistances(Tensor a, Tensor b) {
        Tensor aa = a.square().sum(1, true);
        Tensor bb = b.square().sum(1, false).unsqueeze(0);
        return aa.add(bb).sub(a.matmul(b, false, true).mul(2.0f)).relu();
    }

    /** The distinct labels, sorted. */
    static int[] classes(int[] y) {
        TreeSet<Integer> s = new TreeSet<>();
        for (int v : y) {
            s.add(v);
        }
        return s.stream().mapToInt(Integer::intValue).toArray();
    }

    /** y as indices into classes. */
    static int[] encode(int[] y, int[] classes) {
        int[] out = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = Arrays.binarySearch(classes, y[i]);
            if (out[i] < 0) {
                throw new IllegalArgumentException("label " + y[i] + " was not seen in fit");
            }
        }
        return out;
    }

    static int[] decode(int[] idx, int[] classes) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = classes[idx[i]];
        }
        return out;
    }

    /** One-hot rows [n, c] of class indices. */
    static Tensor oneHot(int[] idx, int c) {
        return F.oneHot(Tensor.of(idx, idx.length), c);
    }

    static void check(Tensor x, String what) {
        if (x.rank() != 2) {
            throw new IllegalArgumentException(what + " needs samples as rows [n, features], got " + x.shape());
        }
    }

    /** Runs what is recorded so far, so that iterative estimators replay one cached plan per step. */
    static void step() {
        Engine.sync();
    }
}
