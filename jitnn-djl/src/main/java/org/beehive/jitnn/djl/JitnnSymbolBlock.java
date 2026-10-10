package org.beehive.jitnn.djl;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import org.beehive.jitnn.DType;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.onnx.OnnxSession;

/**
 * An ONNX model loaded by jitnn-onnx, as a DJL block. Inputs are matched to the graph's inputs by
 * name when the arrays are named after them, else by position; integer arrays (token ids) go in
 * as INT32. The outputs come back in the graph's order.
 */
public final class JitnnSymbolBlock extends AbstractSymbolBlock {

    private final JitnnNDManager manager;
    private OnnxSession session;

    JitnnSymbolBlock(JitnnNDManager manager) {
        this.manager = manager;
    }

    JitnnSymbolBlock(JitnnNDManager manager, OnnxSession session) {
        this.manager = manager;
        this.session = session;
        this.inputNames = session.inputNames();
    }

    /** The jitnn-onnx session behind this block. */
    public OnnxSession session() {
        return session;
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        if (session == null) {
            throw new IllegalStateException("no model loaded into this block");
        }
        List<String> names = session.inputNames();
        if (inputs.size() != names.size()) {
            throw new IllegalArgumentException("the model takes " + names.size() + " inputs " + names + ", got " + inputs.size());
        }
        boolean byName = inputs.stream().allMatch(a -> a.getName() != null && names.contains(a.getName()));
        Map<String, Tensor> feed = new HashMap<>();
        for (int i = 0; i < inputs.size(); i++) {
            NDArray a = inputs.get(i);
            JitnnNDArray j = manager.from(a);
            Tensor t = a.getDataType().isInteger() ? JitnnNDArray.ints(j.tensor()) : j.tensor();
            feed.put(byName ? a.getName() : names.get(i), t);
        }
        Map<String, Tensor> out = session.run(feed);
        NDList result = new NDList();
        for (String name : session.outputNames()) {
            Tensor t = out.get(name);
            JitnnNDArray r = t.dtype() == DType.I32 ? manager.wrap(t.to(DType.F32), DataType.INT64) : manager.wrap(t, DataType.FLOAT32);
            r.setName(name);
            result.add(r);
        }
        return result;
    }

    @Override
    public ai.djl.nn.ParameterList getDirectParameters() {
        return new ai.djl.nn.ParameterList();
    }

    @Override
    public PairList<String, Shape> describeInput() {
        PairList<String, Shape> d = new PairList<>();
        if (session != null) {
            for (String n : session.inputNames()) {
                d.add(n, null);
            }
        }
        return d;
    }
}
