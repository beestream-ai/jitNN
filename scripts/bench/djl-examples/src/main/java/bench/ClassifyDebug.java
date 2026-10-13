package bench;

import java.nio.file.Paths;
import java.util.Arrays;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.nn.Parameter;

/** ImageClassification's steps one at a time, with a summary of each intermediate array. */
public final class ClassifyDebug {

    static void stat(String what, NDArray a) {
        float[] v = a.toType(ai.djl.ndarray.types.DataType.FLOAT32, false).toFloatArray();
        double sum = 0;
        for (float x : v) {
            sum += x;
        }
        System.out.printf("[dbg] %-28s %-16s %-8s sum %.6g first %s%n", what, a.getShape(), a.getDataType(), sum,
                Arrays.toString(Arrays.copyOf(v, Math.min(6, v.length))));
    }

    public static void main(String[] args) throws Exception {
        System.out.println("[dbg] engine " + Engine.getDefaultEngineName());
        Image img = ImageFactory.getInstance().fromFile(Paths.get("src/test/resources/0.png"));
        try (Model model = Model.newInstance("mlp", Engine.getDefaultEngineName())) {
            model.setBlock(new Mlp(28 * 28, 10, new int[] { 128, 64 }));
            model.load(Paths.get("build/model"));
            NDManager m = model.getNDManager();
            for (Pair<String, Parameter> p : model.getBlock().getParameters()) {
                stat("param " + p.getKey(), p.getValue().getArray());
            }
            NDArray a = img.toNDArray(m, Image.Flag.COLOR);
            stat("image", a);
            NDArray t = new ToTensor().transform(a);
            stat("toTensor", t);
            NDArray b = t.expandDims(0);
            stat("batched", b);
            NDList out = model.getBlock().forward(new ParameterStore(m, false), new NDList(b), false);
            stat("logits", out.head());
            stat("softmax(0)", out.head().softmax(0));
            NDList un = ai.djl.translate.Batchifier.STACK.unbatchify(out)[0];
            stat("unbatchified", un.head());
            stat("unbatchified softmax(0)", un.head().softmax(0));
        }
        System.exit(0);
    }
}
