package ai.beestream.jitnn;

import java.util.Arrays;

/** An immutable, row-major tensor shape. A rank-0 shape is a scalar with one element. */
public final class Shape {

    private final int[] dims;
    private final int numel;

    private Shape(int[] dims) {
        long n = 1;
        for (int d : dims) {
            if (d < 0) {
                throw new IllegalArgumentException("negative dimension in " + Arrays.toString(dims));
            }
            n *= d;
        }
        if (n > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("tensor too large: " + Arrays.toString(dims));
        }
        this.dims = dims;
        this.numel = (int) n;
    }

    public static Shape of(int... dims) {
        return new Shape(dims.clone());
    }

    public static Shape scalar() {
        return new Shape(new int[0]);
    }

    public int rank() {
        return dims.length;
    }

    /** Dimension {@code axis}; negative values count from the end. */
    public int dim(int axis) {
        return dims[axis(axis)];
    }

    public int[] dims() {
        return dims.clone();
    }

    public int numel() {
        return numel;
    }

    /** Normalises a possibly negative axis to {@code [0, rank)}. */
    public int axis(int axis) {
        int a = axis < 0 ? axis + dims.length : axis;
        if (a < 0 || a >= dims.length) {
            throw new IllegalArgumentException("axis " + axis + " out of range for shape " + this);
        }
        return a;
    }

    /** Contiguous row-major strides, in elements. */
    public int[] strides() {
        int[] s = new int[dims.length];
        int acc = 1;
        for (int i = dims.length - 1; i >= 0; i--) {
            s[i] = acc;
            acc *= dims[i];
        }
        return s;
    }

    /** The shape with {@code axis} removed ({@code keepDim=false}) or set to 1 ({@code keepDim=true}). */
    public Shape reduce(int axis, boolean keepDim) {
        int a = axis(axis);
        if (keepDim) {
            int[] d = dims.clone();
            d[a] = 1;
            return new Shape(d);
        }
        int[] d = new int[dims.length - 1];
        for (int i = 0, j = 0; i < dims.length; i++) {
            if (i != a) {
                d[j++] = dims[i];
            }
        }
        return new Shape(d);
    }

    /** NumPy broadcasting of two shapes. */
    public static Shape broadcast(Shape a, Shape b) {
        int r = Math.max(a.rank(), b.rank());
        int[] out = new int[r];
        for (int i = 0; i < r; i++) {
            int da = i < r - a.rank() ? 1 : a.dims[i - (r - a.rank())];
            int db = i < r - b.rank() ? 1 : b.dims[i - (r - b.rank())];
            if (da != db && da != 1 && db != 1) {
                throw new IllegalArgumentException("shapes " + a + " and " + b + " do not broadcast");
            }
            out[i] = da == 1 ? db : da;
        }
        return new Shape(out);
    }

    /**
     * Strides for reading a tensor of this shape as if it had shape {@code target}: broadcast
     * dimensions get stride 0. Returned with rank {@code target.rank()}.
     */
    public int[] broadcastStrides(Shape target) {
        int r = target.rank();
        int[] own = strides();
        int[] s = new int[r];
        int off = r - dims.length;
        for (int i = 0; i < r; i++) {
            if (i < off) {
                s[i] = 0;
            } else {
                int d = dims[i - off];
                if (d != 1 && d != target.dims[i]) {
                    throw new IllegalArgumentException(this + " does not broadcast to " + target);
                }
                s[i] = d == 1 ? 0 : own[i - off];
            }
        }
        return s;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof Shape s && Arrays.equals(dims, s.dims);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dims);
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
