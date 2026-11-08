package ai.beestream.jitnn.djl;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.index.full.NDIndexFullPick;
import ai.djl.ndarray.index.full.NDIndexFullSlice;
import ai.djl.ndarray.index.full.NDIndexFullTake;
import ai.djl.ndarray.types.Shape;

import ai.beestream.jitnn.Tensor;

/** NDIndex reads and writes on jitNN: slices narrow or select, picks gather, writes scatter. */
final class JitnnIndexer extends NDArrayIndexer {

    private final JitnnNDManager manager;

    JitnnIndexer(JitnnNDManager manager) {
        this.manager = manager;
    }

    @Override
    public NDArray get(NDArray array, NDIndexFullPick fullPick) {
        JitnnNDArray a = manager.from(array);
        int axis = fullPick.getAxis();
        Tensor idx = JitnnNDArray.ints(a.of(fullPick.getIndices()));
        if (idx.rank() < a.tensor().rank()) {
            idx = idx.unsqueeze(axis);
        }
        return a.wrap(a.tensor().gather(axis, idx), a.getDataType());
    }

    @Override
    public NDArray get(NDArray array, NDIndexFullTake fullTake) {
        JitnnNDArray a = manager.from(array);
        int axis = fullTake.getAxis();
        Tensor idx = JitnnNDArray.ints(a.of(fullTake.getIndices()));
        Tensor t = a.tensor();
        Tensor picked = t.indexSelect(axis, idx.reshape(-1));
        int[] d = t.shape().dims();
        int[] id = idx.shape().dims();
        int[] out = new int[d.length - 1 + id.length];
        int k = 0;
        for (int i = 0; i < axis; i++) {
            out[k++] = d[i];
        }
        for (int v : id) {
            out[k++] = v;
        }
        for (int i = axis + 1; i < d.length; i++) {
            out[k++] = d[i];
        }
        return a.wrap(picked.reshape(out), a.getDataType());
    }

    @Override
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        JitnnNDArray a = manager.from(array);
        Tensor t = a.tensor();
        long[] min = fullSlice.getMin();
        long[] step = fullSlice.getStep();
        Shape region = fullSlice.getShape();
        for (int axis = 0; axis < min.length; axis++) {
            int len = (int) region.get(axis);
            if (min[axis] == 0 && step[axis] == 1 && len == t.dim(axis)) {
                continue;
            }
            if (step[axis] == 1 && min[axis] == 0) {
                t = t.narrow(axis, 0, len);
            } else {
                // a moving window (a dataset read batch by batch) as an index list, a runtime input:
                // as narrow's start the offset would be part of the plan, recompiling every batch
                int[] pos = new int[len];
                for (int i = 0; i < len; i++) {
                    pos[i] = (int) (min[axis] + i * step[axis]);
                }
                t = t.indexSelect(axis, Tensor.of(pos, len));
            }
        }
        return a.wrap(t.reshape(JitnnNDArray.dims(fullSlice.getSqueezedShape())), a.getDataType());
    }

    /** Flat positions of a slice region, in row-major order. */
    private static int[] positions(NDIndexFullSlice slice, Shape full) {
        long[] min = slice.getMin();
        long[] step = slice.getStep();
        Shape region = slice.getShape();
        int rank = full.dimension();
        int count = Math.toIntExact(region.size());
        int[] out = new int[count];
        int[] coord = new int[rank];
        long[] strides = new long[rank];
        long s = 1;
        for (int d = rank - 1; d >= 0; d--) {
            strides[d] = s;
            s *= full.get(d);
        }
        for (int i = 0; i < count; i++) {
            long flat = 0;
            for (int d = 0; d < rank; d++) {
                flat += (min[d] + coord[d] * step[d]) * strides[d];
            }
            out[i] = (int) flat;
            for (int d = rank - 1; d >= 0; d--) {
                if (++coord[d] < region.get(d)) {
                    break;
                }
                coord[d] = 0;
            }
        }
        return out;
    }

    private void write(NDArray array, NDIndexFullSlice slice, Tensor value) {
        JitnnNDArray a = manager.from(array);
        Tensor t = a.tensor();
        int[] pos = positions(slice, array.getShape());
        int[] region = JitnnNDArray.dims(slice.getShape());
        Tensor v = value.numel() == pos.length ? value.reshape(region)
                : value.expand(ai.beestream.jitnn.Shape.of(JitnnNDArray.dims(slice.getSqueezedShape()))).reshape(region);
        Tensor out = t.reshape(-1).scatter(0, Tensor.of(pos, pos.length), v.reshape(-1)).reshape(t.shape().dims());
        ((JitnnNDArray) array).replace(a.wrap(out, a.getDataType()));
    }

    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        write(array, fullSlice, manager.from(value).tensor());
    }

    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        int n = Math.toIntExact(fullSlice.getShape().size());
        write(array, fullSlice, Tensor.full(value.floatValue(), n));
    }
}
