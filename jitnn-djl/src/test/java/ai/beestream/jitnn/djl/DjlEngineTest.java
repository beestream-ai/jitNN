package ai.beestream.jitnn.djl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.NoopTranslator;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.onnx.OnnxSession;
import org.junit.jupiter.api.Test;

/** DJL code on jitNN's host reference: NDArray semantics, training with DJL's Trainer, ONNX models. */
class DjlEngineTest {

    @Test
    void isTheDefaultEngine() {
        assertEquals("jitNN", Engine.getInstance().getEngineName());
    }

    @Test
    void arrays() {
        try (NDManager m = NDManager.newBaseManager()) {
            NDArray a = m.create(new float[] { 1, 2, 3, 4, 5, 6 }, new Shape(2, 3));
            NDArray b = m.create(new float[] { 10, 20, 30 });
            assertArrayEquals(new float[] { 11, 22, 33, 14, 25, 36 }, a.add(b).toFloatArray());
            assertArrayEquals(new float[] { 6, 15 }, a.sum(new int[] { 1 }).toFloatArray());
            assertArrayEquals(new float[] { 2.5f, 3.5f, 4.5f }, a.mean(new int[] { 0 }).toFloatArray());
            assertEquals(21f, a.sum().getFloat());
            assertArrayEquals(new float[] { 140, 320 }, a.matMul(b.reshape(3, 1)).toFloatArray());
            assertArrayEquals(new float[] { 1, 4, 2, 5, 3, 6 }, a.transpose().toFloatArray());
            assertArrayEquals(new float[] { 2, 3, 5, 6 }, a.get(":, 1:").toFloatArray());
            assertArrayEquals(new float[] { 4, 5, 6 }, a.get(1).toFloatArray());
            assertArrayEquals(new float[] { 1, 3, 4, 6 }, a.get(":, ::2").toFloatArray());
            assertArrayEquals(new long[] { 2, 2 }, a.argMax(1).toLongArray());
            assertEquals(DataType.INT64, a.argMax(1).getDataType());
            assertArrayEquals(new boolean[] { false, false, true, true, true, true }, a.gt(2).toBooleanArray());
            NDArray s = a.softmax(1);
            assertEquals(1.0, s.sum(new int[] { 1 }).toFloatArray()[0], 1e-6);
            assertArrayEquals(new float[] { 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6 }, NDArrays.concat(new NDList(a, a)).toFloatArray());
            assertEquals(new Shape(2, 2, 3), NDArrays.stack(new NDList(a, a)).getShape());
            NDList parts = a.split(3, 1);
            assertArrayEquals(new float[] { 2, 5 }, parts.get(1).toFloatArray());
            NDArray c = a.duplicate();
            c.set(new NDIndex("0, :"), 0f);
            assertArrayEquals(new float[] { 0, 0, 0, 4, 5, 6 }, c.toFloatArray());
            assertArrayEquals(new float[] { 1, 2, 3, 4, 5, 6 }, a.toFloatArray());
            c.addi(1);
            assertArrayEquals(new float[] { 1, 1, 1, 5, 6, 7 }, c.toFloatArray());
            NDArray ints = m.create(new int[] { 7, -7 });
            assertArrayEquals(new int[] { 3, -3 }, ints.div(2).toIntArray());
            assertArrayEquals(new int[] { 1, 1 }, ints.mod(2).toIntArray());
            // a floating scalar promotes integer arrays, as DJL's ToTensor needs for uint8 images
            assertArrayEquals(new float[] { 3.5f, -3.5f }, ints.div(2.0).toFloatArray());
            NDArray pixels = m.create(java.nio.ByteBuffer.wrap(new byte[] { 0, (byte) 128, (byte) 255 }), new Shape(3), DataType.UINT8);
            assertEquals(DataType.FLOAT32, pixels.div(255.0).getDataType());
            assertArrayEquals(new float[] { 0, 128 / 255f, 1 }, pixels.div(255.0).toFloatArray(), 1e-6f);
            assertEquals(DataType.UINT8, pixels.div(2).getDataType());
            assertArrayEquals(new float[] { 2, 5 }, a.get(new NDIndex(":, {}", m.create(new int[] { 1, 2 }))).get(":, 0").toFloatArray());
        }
    }

    @Test
    void trainsAnMlp() {
        List<Float> losses = DjlWorkloads.trainMlp(60, Optimizer.adam().optLearningRateTracker(Tracker.fixed(0.02f)).build());
        assertTrue(losses.get(losses.size() - 1) < 0.1f * losses.get(0), "loss " + losses.get(0) + " -> " + losses.get(losses.size() - 1));
    }

    @Test
    void trainsACnnWithBatchNorm() {
        List<Float> losses = DjlWorkloads.trainCnn(30);
        assertTrue(losses.get(losses.size() - 1) < 0.5f * losses.get(0), "loss " + losses.get(0) + " -> " + losses.get(losses.size() - 1));
    }

    @Test
    void trainsAnLstm() {
        List<Float> losses = DjlWorkloads.trainLstm(40);
        assertTrue(losses.get(losses.size() - 1) < 0.5f * losses.get(0), "loss " + losses.get(0) + " -> " + losses.get(losses.size() - 1));
    }

    @Test
    void loadsOnnxModels() throws Exception {
        Path file = Path.of("../jitnn-onnx/src/test/resources/onnx/torch_cnn.onnx");
        OnnxSession direct = OnnxSession.load(file);
        String in = direct.inputNames().get(0);
        float[] image = Tensor.uniform(5, -1, 1, 2, 3, 16, 16).toFloatArray();
        float[] expected = direct.run(Map.of(in, Tensor.of(image, 2, 3, 16, 16))).values().iterator().next().toFloatArray();
        try (Model model = Model.newInstance("cnn")) {
            model.load(file);
            try (Predictor<NDList, NDList> p = model.newPredictor(new NoopTranslator())) {
                NDList out = p.predict(new NDList(model.getNDManager().create(image, new Shape(2, 3, 16, 16))));
                assertArrayEquals(expected, out.head().toFloatArray(), 1e-6f);
            }
        }
    }
}
