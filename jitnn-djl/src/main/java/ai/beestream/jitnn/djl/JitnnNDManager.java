/*
 * Copyright (c) 2026 Beestream Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.beestream.jitnn.djl;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import ai.beestream.jitnn.JitnnConfig;
import ai.beestream.jitnn.Tensor;

/**
 * Creates {@link JitnnNDArray}s. Every numeric DJL type is held in an FP32 jitNN tensor (exact for
 * integers up to 2^24) tagged with its DJL type, which arithmetic results and host copies follow;
 * operators that take indices convert to INT32 on the device.
 */
public class JitnnNDManager extends BaseNDManager {

    private static final JitnnNDManager SYSTEM = new SystemManager();

    JitnnNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static JitnnNDManager getSystemManager() {
        return SYSTEM;
    }

    @Override
    public Engine getEngine() {
        return JitnnEngine.getInstance();
    }

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    @Override
    public JitnnNDArray from(NDArray array) {
        if (array == null || array instanceof JitnnNDArray) {
            return (JitnnNDArray) array;
        }
        JitnnNDArray result = create(array.toByteBuffer(), array.getShape(), array.getDataType());
        result.setName(array.getName());
        return result;
    }

    @Override
    public NDManager newSubManager(Device device) {
        JitnnNDManager manager = new JitnnNDManager(this, device);
        attachUncappedInternal(manager.uid, manager);
        return manager;
    }

    /** The fallback engine's manager, or null. */
    NDManager alternative() {
        return alternativeManager;
    }

    /** Wraps a jitNN tensor as an array of this manager. */
    public JitnnNDArray wrap(Tensor tensor, DataType dataType) {
        return new JitnnNDArray(this, tensor, dataType);
    }

    // ---------------------------------------------------------------- creation

    @Override
    public JitnnNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        float[] values = new float[size];
        Buffer b = data.duplicate();
        b.rewind();
        if (b instanceof ByteBuffer bytes) {
            bytes.order(ByteOrder.nativeOrder());
            b = dataType.asDataType(bytes);
        }
        if (b instanceof FloatBuffer f) {
            f.get(values, 0, size);
        } else if (b instanceof DoubleBuffer d) {
            for (int i = 0; i < size; i++) {
                values[i] = (float) d.get(i);
            }
        } else if (b instanceof IntBuffer in) {
            for (int i = 0; i < size; i++) {
                values[i] = dataType == DataType.UINT32 ? (float) Integer.toUnsignedLong(in.get(i)) : in.get(i);
            }
        } else if (b instanceof LongBuffer l) {
            for (int i = 0; i < size; i++) {
                values[i] = l.get(i);
            }
        } else if (b instanceof ShortBuffer s) {
            for (int i = 0; i < size; i++) {
                short v = s.get(i);
                values[i] = switch (dataType) {
                    case FLOAT16 -> Float.float16ToFloat(v);
                    case BFLOAT16 -> Float.intBitsToFloat((v & 0xFFFF) << 16);
                    case UINT16 -> v & 0xFFFF;
                    default -> v;
                };
            }
        } else if (b instanceof ByteBuffer bytes) {
            for (int i = 0; i < size; i++) {
                byte v = bytes.get(i);
                values[i] = dataType == DataType.INT8 ? v : v & 0xFF;
            }
        } else {
            throw new IllegalArgumentException("jitNN arrays cannot be created from " + data.getClass().getSimpleName() + " as " + dataType);
        }
        Tensor t = Tensor.of(values, JitnnNDArray.dims(shape));
        // large arrays (datasets) stay on the device: an ordinary host tensor is uploaded by every
        // step that reads it, a persistent one once
        return wrap((long) size * 4 >= PERSISTENT_BYTES ? Tensor.persistent(t) : t, dataType);
    }

    /** Arrays created from host data at least this large are kept in persistent device storage. */
    static final long PERSISTENT_BYTES = JitnnConfig.current().get(JitnnConfig.DJL_PERSISTENT_BYTES);

    @Override
    public JitnnNDArray create(Shape shape, DataType dataType) {
        return zeros(shape, dataType);
    }

    @Override
    public JitnnNDArray zeros(Shape shape, DataType dataType) {
        return wrap(host(new float[Math.toIntExact(shape.size())], shape), dataType);
    }

    @Override
    public JitnnNDArray ones(Shape shape, DataType dataType) {
        return full(shape, 1, dataType);
    }

    @Override
    public JitnnNDArray full(Shape shape, float value, DataType dataType) {
        float[] v = new float[Math.toIntExact(shape.size())];
        java.util.Arrays.fill(v, value);
        return wrap(host(v, shape), dataType);
    }

    @Override
    public JitnnNDArray arange(float start, float stop, float step, DataType dataType) {
        int n = (int) Math.max(0, Math.ceil((stop - start) / step));
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = start + i * step;
        }
        return wrap(Tensor.of(v, n), dataType);
    }

    @Override
    public JitnnNDArray eye(int rows, int cols, int k, DataType dataType) {
        return wrap(onHost(() -> Tensor.eye(rows, cols, k)), dataType);
    }

    @Override
    public JitnnNDArray linspace(float start, float stop, int num, boolean endpoint) {
        float[] v = new float[num];
        double step = num <= 1 ? 0 : (stop - start) / (double) (endpoint ? num - 1 : num);
        for (int i = 0; i < num; i++) {
            v[i] = (float) (start + i * step);
        }
        return wrap(Tensor.of(v, num), DataType.FLOAT32);
    }

    /**
     * Created arrays hold host data, never a device intermediate: DJL keeps them (parameters,
     * running statistics) across steps, and an intermediate's buffer belongs to its cached plan,
     * which an identical later creation replays.
     */
    private static Tensor host(float[] values, Shape shape) {
        Tensor t = Tensor.of(values, JitnnNDArray.dims(shape));
        return (long) values.length * 4 >= PERSISTENT_BYTES ? Tensor.persistent(t) : t;
    }

    /** Computes {@code f} with jitNN's host reference (same values as the devices: Philox, exact ops) and returns host data. */
    private static Tensor onHost(java.util.function.Supplier<Tensor> f) {
        float[][] out = new float[1][];
        int[][] dims = new int[1][];
        ai.beestream.jitnn.engine.Engine.with(ai.beestream.jitnn.engine.Engine.Backend.REFERENCE, () -> {
            Tensor t = f.get();
            out[0] = t.toFloatArray();
            dims[0] = t.shape().dims();
            return null;
        });
        Tensor t = Tensor.of(out[0], dims[0]);
        return (long) out[0].length * 4 >= PERSISTENT_BYTES ? Tensor.persistent(t) : t;
    }

    private long seed() {
        return JitnnEngine.getInstance().nextSeed();
    }

    @Override
    public JitnnNDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        long seed = seed();
        return wrap(onHost(() -> Tensor.uniform(seed, low, high, JitnnNDArray.dims(shape))), dataType);
    }

    @Override
    public JitnnNDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        long seed = seed();
        return wrap(onHost(() -> Tensor.normal(seed, loc, scale, JitnnNDArray.dims(shape))), dataType);
    }

    @Override
    public JitnnNDArray truncatedNormal(float loc, float scale, Shape shape, DataType dataType) {
        // inverse CDF of the normal truncated to [-2, 2] standard deviations, from a uniform draw
        double lo = 0.5 * (1 + erf(-2 / Math.sqrt(2)));
        double hi = 0.5 * (1 + erf(2 / Math.sqrt(2)));
        long seed = seed();
        return wrap(onHost(() -> {
            Tensor u = Tensor.uniform(seed, (float) lo, (float) hi, JitnnNDArray.dims(shape));
            return JitnnNDArray.erfinv(u.mul(2).add(-1)).mul((float) Math.sqrt(2)).mul(scale).add(loc);
        }), dataType);
    }

    private static double erf(double x) {
        // Abramowitz and Stegun 7.1.26 is enough for the two constants above
        double t = 1 / (1 + 0.3275911 * Math.abs(x));
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
        return Math.signum(x) * y;
    }

    @Override
    public JitnnNDArray randomInteger(long low, long high, Shape shape, DataType dataType) {
        long seed = seed();
        return wrap(onHost(() -> Tensor.uniform(seed, low, high, JitnnNDArray.dims(shape)).floor().clamp(low, high - 1)), dataType);
    }

    @Override
    public JitnnNDArray randomPermutation(long n) {
        long seed = seed();
        return wrap(onHost(() -> Tensor.uniform(seed, 0, 1, Math.toIntExact(n)).argsort(0, false).to(ai.beestream.jitnn.DType.F32)), DataType.INT64);
    }

    /** The root manager: never closed, holds detached arrays. */
    private static final class SystemManager extends JitnnNDManager implements SystemNDManager {

        SystemManager() {
            super(null, null);
        }

        @Override
        public void attachInternal(String resourceId, AutoCloseable... resources) {
        }

        @Override
        public void attachUncappedInternal(String resourceId, AutoCloseable resource) {
        }

        @Override
        public void tempAttachInternal(NDManager originalManager, String resourceId, ai.djl.ndarray.NDResource resource) {
        }

        @Override
        public void detachInternal(String resourceId) {
        }

        @Override
        public void close() {
        }
    }
}
