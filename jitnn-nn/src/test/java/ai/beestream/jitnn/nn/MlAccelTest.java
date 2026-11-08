package ai.beestream.jitnn.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.engine.Engine;
import ai.beestream.jitnn.ml.GradientBoosting;
import ai.beestream.jitnn.ml.KMeans;
import ai.beestream.jitnn.ml.KNeighbors;
import ai.beestream.jitnn.ml.LinearRegression;
import ai.beestream.jitnn.ml.LinearSvc;
import ai.beestream.jitnn.ml.LogisticRegression;
import ai.beestream.jitnn.ml.NaiveBayes;
import ai.beestream.jitnn.ml.Pca;
import ai.beestream.jitnn.ml.RandomForest;
import ai.beestream.jitnn.ml.Tsne;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The classical machine-learning estimators against scikit-learn (references from
 * scripts/sklearn/ml_goldens.py in src/test/resources/ml): exact algorithms must reproduce
 * scikit-learn's results, iterative ones its optimum, and tree ensembles and t-SNE its held-out
 * quality within a small margin.
 */
class MlAccelTest {

    record Arr(int[] dims, float[] f, int[] i) {
    }

    static Map<String, Map<String, Arr>> cases;

    @BeforeAll
    static void load() throws IOException {
        Path dir = Path.of("src", "test", "resources", "ml");
        ByteBuffer data = ByteBuffer.wrap(Files.readAllBytes(dir.resolve("data.bin"))).order(ByteOrder.LITTLE_ENDIAN);
        cases = new HashMap<>();
        for (String line : Files.readAllLines(dir.resolve("manifest.txt"))) {
            String[] t = line.trim().split("\\s+");
            int offset = Integer.parseInt(t[3]);
            int count = Integer.parseInt(t[4]);
            int[] dims = new int[t.length - 5];
            for (int k = 0; k < dims.length; k++) {
                dims[k] = Integer.parseInt(t[5 + k]);
            }
            Arr a;
            if (t[2].equals("i32")) {
                int[] v = new int[count];
                for (int k = 0; k < count; k++) {
                    v[k] = data.getInt(offset + 4 * k);
                }
                a = new Arr(dims, null, v);
            } else {
                float[] v = new float[count];
                for (int k = 0; k < count; k++) {
                    v[k] = data.getFloat(offset + 4 * k);
                }
                a = new Arr(dims, v, null);
            }
            cases.computeIfAbsent(t[0], k -> new HashMap<>()).put(t[1], a);
        }
    }

    @AfterAll
    static void release() {
        Engine.reset();
    }

    static Arr get(String c, String name) {
        return cases.get(c).get(name);
    }

    static Tensor tensor(String c, String name) {
        Arr a = get(c, name);
        return a.f != null ? Tensor.of(a.f, a.dims) : Tensor.of(a.i, a.dims);
    }

    static float[] floats(String c, String name) {
        return get(c, name).f;
    }

    static int[] ints(String c, String name) {
        return get(c, name).i;
    }

    static float scalar(String c, String name) {
        return get(c, name).f[0];
    }

    /** max |a - e| <= tol * max(1, max |e|). */
    static void close(String what, float[] e, float[] a, double tol) {
        assertTrue(e.length == a.length, what + ": " + e.length + " expected values, got " + a.length);
        double scale = 1;
        double worst = 0;
        int at = -1;
        for (int i = 0; i < e.length; i++) {
            scale = Math.max(scale, Math.abs(e[i]));
        }
        for (int i = 0; i < e.length; i++) {
            double d = Math.abs(e[i] - a[i]);
            if (d > worst) {
                worst = d;
                at = i;
            }
        }
        System.out.printf("[ml] %-28s max error %.2e (scale %.2f)%n", what, worst, scale);
        assertTrue(worst <= tol * scale, what + ": element " + at + " expected " + (at >= 0 ? e[at] : 0) + " got " + (at >= 0 ? a[at] : 0) + " (error " + worst + ")");
    }

    static float[] centred(float[] v, int rows) {
        float[] out = v.clone();
        double m = 0;
        for (float x : v) {
            m += x;
        }
        m /= rows;
        for (int i = 0; i < out.length; i++) {
            out[i] -= (float) m;
        }
        return out;
    }

    static float accuracy(int[] a, int[] b) {
        int ok = 0;
        for (int i = 0; i < a.length; i++) {
            ok += a[i] == b[i] ? 1 : 0;
        }
        return (float) ok / a.length;
    }

    static float r2(float[] y, float[] p) {
        double mean = 0;
        for (float v : y) {
            mean += v;
        }
        mean /= y.length;
        double ssr = 0;
        double sst = 0;
        for (int i = 0; i < y.length; i++) {
            ssr += (y[i] - p[i]) * (y[i] - p[i]);
            sst += (y[i] - mean) * (y[i] - mean);
        }
        return (float) (1 - ssr / sst);
    }

    @Test
    void kmeans() {
        KMeans km = new KMeans(3, 300, 1e-4f, 1, 0).fit(tensor("kmeans", "x"), tensor("kmeans", "init"));
        close("kmeans centres", floats("kmeans", "centres"), km.centres().toFloatArray(), 1e-4);
        assertArrayEquals(ints("kmeans", "labels"), km.predict(tensor("kmeans", "x")), "kmeans labels");
        close("kmeans inertia", floats("kmeans", "inertia"), new float[] { km.inertia() }, 1e-4);
        KMeans pp = new KMeans(5, 300, 1e-4f, 10, 0).fit(tensor("kmeans_pp", "x"));
        float sk = scalar("kmeans_pp", "inertia");
        System.out.printf("[ml] kmeans++ inertia %.3f (scikit-learn %.3f)%n", pp.inertia(), sk);
        assertTrue(pp.inertia() <= 1.01f * sk, "kmeans++ inertia " + pp.inertia() + " vs scikit-learn " + sk);
    }

    @Test
    void neighbours() {
        KNeighbors knn = new KNeighbors(5).fit(tensor("knn", "x"), ints("knn", "y"));
        assertArrayEquals(ints("knn", "pred"), knn.predict(tensor("knn", "q")), "kNN classes");
        close("kNN probabilities", floats("knn", "proba"), knn.predictProba(tensor("knn", "q")).toFloatArray(), 1e-6);
        KNeighbors reg = new KNeighbors(5).fit(tensor("knn", "x"), tensor("knn", "target"));
        close("kNN regression", floats("knn", "reg"), reg.predictValues(tensor("knn", "q")).toFloatArray(), 1e-5);
    }

    @Test
    void pca() {
        Pca pca = new Pca(3).fit(tensor("pca", "x"));
        close("PCA components", floats("pca", "components"), pca.components().toFloatArray(), 1e-3);
        close("PCA variance", floats("pca", "variance"), pca.explainedVariance(), 1e-3);
        close("PCA variance ratio", floats("pca", "ratio"), pca.explainedVarianceRatio(), 1e-4);
        close("PCA transform", floats("pca", "transformed"), pca.transform(tensor("pca", "x")).toFloatArray(), 1e-3);
    }

    @Test
    void linear() {
        LinearRegression lr = new LinearRegression().fit(tensor("linear", "x"), tensor("linear", "y"));
        close("least squares coef", floats("linear", "coef"), lr.coef().toFloatArray(), 1e-4);
        close("least squares intercept", floats("linear", "intercept"), lr.intercept().toFloatArray(), 1e-4);
        LinearRegression rd = new LinearRegression(5.0f, true).fit(tensor("linear", "x"), tensor("linear", "y"));
        close("ridge coef", floats("linear", "ridge_coef"), rd.coef().toFloatArray(), 1e-4);
        close("ridge intercept", floats("linear", "ridge_intercept"), rd.intercept().toFloatArray(), 1e-4);
    }

    @Test
    void logistic() {
        for (String c : new String[] { "logreg_binary", "logreg_multi" }) {
            LogisticRegression m = new LogisticRegression(1.0f, 1000).fit(tensor(c, "x"), ints(c, "y"));
            close(c + " coef", floats(c, "coef"), m.coef().toFloatArray(), 1e-3);
            int rows = get(c, "intercept").dims[0];
            close(c + " intercept", centred(floats(c, "intercept"), rows), centred(m.intercept().toFloatArray(), rows), 1e-3);
            close(c + " probabilities", floats(c, "proba"), m.predictProba(tensor(c, "x")).toFloatArray(), 1e-3);
        }
    }

    @Test
    void svm() {
        LinearSvc svc = new LinearSvc(1.0f, 2000).fit(tensor("svc", "x"), ints("svc", "y"));
        close("linear SVM coef", floats("svc", "coef"), svc.coef().toFloatArray(), 2e-3);
        close("linear SVM intercept", floats("svc", "intercept"), svc.intercept().toFloatArray(), 2e-3);
        float agree = accuracy(ints("svc", "pred"), svc.predict(tensor("svc", "x")));
        assertTrue(agree >= 0.99f, "linear SVM agrees with scikit-learn on " + agree);
    }

    @Test
    void naiveBayes() {
        NaiveBayes g = new NaiveBayes(NaiveBayes.Kind.GAUSSIAN, 1e-9f).fit(tensor("gaussian_nb", "x"), ints("gaussian_nb", "y"));
        close("Gaussian NB means", floats("gaussian_nb", "theta"), g.theta().toFloatArray(), 1e-5);
        close("Gaussian NB variances", floats("gaussian_nb", "var"), g.var().toFloatArray(), 1e-4);
        close("Gaussian NB probabilities", floats("gaussian_nb", "proba"), g.predictProba(tensor("gaussian_nb", "x")).toFloatArray(), 1e-4);
        NaiveBayes m = new NaiveBayes(NaiveBayes.Kind.MULTINOMIAL, 1.0f).fit(tensor("multinomial_nb", "x"), ints("multinomial_nb", "y"));
        close("multinomial NB log-probs", floats("multinomial_nb", "log_prob"), m.featureLogProb().toFloatArray(), 1e-5);
        close("multinomial NB probabilities", floats("multinomial_nb", "proba"), m.predictProba(tensor("multinomial_nb", "x")).toFloatArray(), 1e-4);
    }

    private static void atLeast(String what, float ours, float sk, float margin) {
        System.out.printf("[ml] %-28s %.4f (scikit-learn %.4f)%n", what, ours, sk);
        assertTrue(ours >= sk - margin, what + ": " + ours + " vs scikit-learn " + sk);
    }

    @Test
    void treesClassification() {
        String c = "trees_classification";
        Tensor x = tensor(c, "x");
        Tensor xt = tensor(c, "xt");
        int[] y = ints(c, "y");
        int[] yt = ints(c, "yt");
        atLeast("decision tree accuracy", accuracy(yt, RandomForest.tree(RandomForest.Task.CLASSIFICATION, 6, 1).fit(x, y).predict(xt)), scalar(c, "tree"), 0.03f);
        atLeast("random forest accuracy", accuracy(yt, new RandomForest(RandomForest.Task.CLASSIFICATION, 100, 8, 0).fit(x, y).predict(xt)), scalar(c, "forest"), 0.03f);
        atLeast("boosting accuracy", accuracy(yt, new GradientBoosting(true, 100, 0.1f, 4).fit(x, y).predict(xt)), scalar(c, "boost"), 0.03f);
    }

    @Test
    void treesRegression() {
        String c = "trees_regression";
        Tensor x = tensor(c, "x");
        Tensor xt = tensor(c, "xt");
        Tensor y = tensor(c, "y");
        float[] yt = floats(c, "yt");
        atLeast("decision tree R2", r2(yt, RandomForest.tree(RandomForest.Task.REGRESSION, 6, 1).fit(x, y).predictValues(xt).toFloatArray()), scalar(c, "tree"), 0.05f);
        atLeast("random forest R2", r2(yt, new RandomForest(RandomForest.Task.REGRESSION, 100, 8, 0).fit(x, y).predictValues(xt).toFloatArray()), scalar(c, "forest"), 0.05f);
        atLeast("boosting R2", r2(yt, new GradientBoosting(false, 100, 0.1f, 4).fit(x, y).predictValues(xt).toFloatArray()), scalar(c, "boost"), 0.05f);
    }

    /** scikit-learn's trustworthiness: how far embedding neighbours were from each other in the input. */
    static float trustworthiness(float[] x, int d, float[] e, int de, int k) {
        int n = x.length / d;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            Integer[] byInput = order(x, d, i, n);
            int[] rank = new int[n];
            for (int r = 0; r < n; r++) {
                rank[byInput[r]] = r;
            }
            Integer[] byEmb = order(e, de, i, n);
            for (int r = 1; r <= k; r++) {
                int j = byEmb[r];
                sum += Math.max(0, rank[j] - k);
            }
        }
        return (float) (1 - 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0)) * sum);
    }

    private static Integer[] order(float[] v, int d, int i, int n) {
        double[] dist = new double[n];
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int c = 0; c < d; c++) {
                double t = v[i * d + c] - v[j * d + c];
                s += t * t;
            }
            dist[j] = j == i ? -1 : s;
        }
        Integer[] o = new Integer[n];
        Arrays.setAll(o, j -> j);
        Arrays.sort(o, (a, b) -> Double.compare(dist[a], dist[b]));
        return o;
    }

    @Test
    void tsne() {
        Tensor x = tensor("tsne", "x");
        float[] emb = new Tsne(2, 20.0f, 1000).fitTransform(x).toFloatArray();
        float t = trustworthiness(floats("tsne", "x"), 10, emb, 2, 5);
        atLeast("t-SNE trustworthiness", t, scalar("tsne", "trust"), 0.02f);
        assertTrue(t >= 0.9f, "t-SNE trustworthiness " + t);
    }
}
