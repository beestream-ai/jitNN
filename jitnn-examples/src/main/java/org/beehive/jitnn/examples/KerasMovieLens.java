package org.beehive.jitnn.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.beehive.jitnn.F;
import org.beehive.jitnn.Grad;
import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.nn.Embedding;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.optim.Adam;

/**
 * Recommends films, after Keras' "collaborative filtering for movie recommendations": 100,836
 * ratings of 9,742 films by 610 MovieLens users. Every user and film gets a learned 50-dimensional
 * vector and a bias; a rating is predicted from their dot product. Afterwards the films a user has
 * not seen are ranked for them. Data: GroupLens ml-latest-small, downloaded on first use.
 *
 * <pre>run.sh KerasMovieLens [epochs] [user id]</pre>
 */
public final class KerasMovieLens {

    /** sigmoid(user . movie + user bias + movie bias), a rating scaled to [0, 1]. */
    static final class RecommenderNet extends Module {

        final Embedding users;
        final Embedding movies;
        final Embedding userBias;
        final Embedding movieBias;

        RecommenderNet(int userCount, int movieCount, int dim) {
            // He-normal vectors and zero biases, as the Keras example initialises them
            float std = (float) Math.sqrt(2.0 / dim);
            this.users = child(new Embedding(Tensor.normal(1, 0, std, userCount, dim), -1));
            this.movies = child(new Embedding(Tensor.normal(2, 0, std, movieCount, dim), -1));
            this.userBias = child(new Embedding(Tensor.zeros(userCount, 1), -1));
            this.movieBias = child(new Embedding(Tensor.zeros(movieCount, 1), -1));
        }

        /** Logits [B, 1] for INT32 user and movie indices [B]. */
        Tensor score(Tensor u, Tensor m) {
            Tensor dot = users.forward(u).mul(movies.forward(m)).sum(1, true);
            return dot.add(userBias.forward(u)).add(movieBias.forward(m));
        }

        @Override
        public Tensor forward(Tensor x) {
            throw new UnsupportedOperationException("use score(users, movies)");
        }
    }

    public static void main(String[] args) throws Exception {
        int epochs = Examples.intArg(args, 0, 5);
        Path dir = movieLens();
        Map<String, Integer> userIndex = new HashMap<>();
        Map<String, Integer> movieIndex = new HashMap<>();
        List<int[]> pairs = new ArrayList<>();
        List<Float> ratings = new ArrayList<>();
        for (String[] f : Examples.csv(dir.resolve("ratings.csv"), ",")) {
            pairs.add(new int[] { userIndex.computeIfAbsent(f[0], k -> userIndex.size()), movieIndex.computeIfAbsent(f[1], k -> movieIndex.size()) });
            ratings.add(Float.parseFloat(f[2]));
        }
        Map<String, String> titles = new HashMap<>();
        for (String line : Files.readAllLines(dir.resolve("movies.csv")).subList(1, Files.readAllLines(dir.resolve("movies.csv")).size())) {
            int a = line.indexOf(',');
            int b = line.lastIndexOf(',');
            titles.put(line.substring(0, a), line.substring(a + 1, b).replace("\"", ""));
        }
        String[] movieId = new String[movieIndex.size()];
        movieIndex.forEach((id, i) -> movieId[i] = id);

        // ratings 0.5 .. 5 to [0, 1]; 90% train, 10% validation, shuffled
        int n = pairs.size();
        int[] order = Examples.permutation(n, 42);
        int[] us = new int[n];
        int[] ms = new int[n];
        float[] ys = new float[n];
        for (int i = 0; i < n; i++) {
            us[i] = pairs.get(order[i])[0];
            ms[i] = pairs.get(order[i])[1];
            ys[i] = (ratings.get(order[i]) - 0.5f) / 4.5f;
        }
        int trainN = n * 9 / 10;
        System.out.printf("%,d ratings by %d users of %,d films%n", n, userIndex.size(), movieIndex.size());

        RecommenderNet net = new RecommenderNet(userIndex.size(), movieIndex.size(), 50);
        Adam opt = new Adam(net.parameters(), 1e-3f, 0.9f, 0.999f, 1e-7f, 1e-6f, false);
        int batch = 64;
        int steps = trainN / batch;
        for (int epoch = 0; epoch < epochs; epoch++) {
            long t0 = System.nanoTime();
            float sum = 0;
            for (int s = 0; s < steps; s++) {
                Tensor u = Tensor.of(Arrays.copyOfRange(us, s * batch, (s + 1) * batch), batch);
                Tensor m = Tensor.of(Arrays.copyOfRange(ms, s * batch, (s + 1) * batch), batch);
                Tensor y = Tensor.of(Arrays.copyOfRange(ys, s * batch, (s + 1) * batch), batch, 1);
                Tensor loss = F.binaryCrossEntropyWithLogits(net.score(u, m), y);
                loss.backward();
                opt.step();
                sum += loss.item();
                opt.zeroGrad();
            }
            // validation error in stars
            int valN = n - trainN;
            float[] p = Grad.inference(() -> net.score(Tensor.of(Arrays.copyOfRange(us, trainN, n), valN), Tensor.of(Arrays.copyOfRange(ms, trainN, n), valN)).sigmoid()).toFloatArray();
            double se = 0;
            for (int i = 0; i < valN; i++) {
                double d = (p[i] - ys[trainN + i]) * 4.5;
                se += d * d;
            }
            System.out.printf("epoch %d: loss %.4f, validation RMSE %.3f stars, %.1f s%n", epoch + 1, sum / steps, Math.sqrt(se / valN), Examples.seconds(t0));
        }

        // the user's best-rated films, and the unseen films the model ranks highest for them
        String user = args.length > 1 ? args[1] : "1";
        int u = userIndex.get(user);
        Set<Integer> seen = new HashSet<>();
        List<int[]> rated = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (pairs.get(i)[0] == u) {
                seen.add(pairs.get(i)[1]);
                rated.add(new int[] { pairs.get(i)[1], Math.round(ratings.get(i) * 2) });
            }
        }
        rated.sort((a, b) -> b[1] - a[1]);
        System.out.printf("%nuser %s rated %d films; their favourites:%n", user, rated.size());
        for (int[] r : rated.subList(0, Math.min(5, rated.size()))) {
            System.out.printf("  %.1f  %s%n", r[1] / 2f, titles.get(movieId[r[0]]));
        }
        int[] candidates = new int[movieIndex.size() - seen.size()];
        int[] sameUser = new int[candidates.length];
        Arrays.fill(sameUser, u);
        for (int m = 0, k = 0; m < movieIndex.size(); m++) {
            if (!seen.contains(m)) {
                candidates[k++] = m;
            }
        }
        float[] scores = Grad.inference(() -> net.score(Tensor.of(sameUser, candidates.length), Tensor.of(candidates, candidates.length))).toFloatArray();
        System.out.println("recommended:");
        for (int i : Examples.topK(scores, 10)) {
            System.out.printf("  %s%n", titles.get(movieId[candidates[i]]));
        }
    }

    /** GroupLens' ml-latest-small, downloaded and unpacked once. */
    static Path movieLens() throws Exception {
        Path dir = Examples.dataDir().resolve("movielens");
        if (!Files.exists(dir.resolve("ml-latest-small").resolve("ratings.csv"))) {
            Examples.unzip(Examples.download("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", dir.resolve("ml-latest-small.zip")), dir);
        }
        return dir.resolve("ml-latest-small");
    }
}
