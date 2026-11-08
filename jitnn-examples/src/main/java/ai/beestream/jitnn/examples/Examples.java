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
package ai.beestream.jitnn.examples;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Helpers shared by the examples: where downloads go ({@code -Djitnn.examples.data}, default
 * {@code ~/.jitnn/examples}), downloading a file once, the standard datasets, and a few numbers.
 */
public final class Examples {

    private Examples() {
    }

    /** The directory the examples download into. */
    public static Path dataDir() {
        String dir = System.getProperty("jitnn.examples.data");
        return dir != null ? Path.of(dir) : Path.of(System.getProperty("user.home"), ".jitnn", "examples");
    }

    /** Downloads {@code url} to {@code file} unless the file is already there; returns the file. */
    public static Path download(String url, Path file) throws IOException {
        if (Files.exists(file)) {
            return file;
        }
        Files.createDirectories(file.getParent());
        System.out.println("downloading " + url);
        Path part = file.resolveSibling(file.getFileName() + ".part");
        HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS).build();
        try {
            HttpResponse<Path> r = http.send(HttpRequest.newBuilder(URI.create(url)).build(), HttpResponse.BodyHandlers.ofFile(part));
            if (r.statusCode() != 200) {
                throw new IOException("HTTP " + r.statusCode() + " for " + url);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
        Files.move(part, file, StandardCopyOption.REPLACE_EXISTING);
        return file;
    }

    /** The MNIST IDX files, downloaded once; the directory to pass to {@code Mnist.load}. */
    public static Path mnist() throws IOException {
        Path dir = dataDir().resolve("mnist");
        for (String f : new String[] { "train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte" }) {
            download("https://ossci-datasets.s3.amazonaws.com/mnist/" + f + ".gz", dir.resolve(f + ".gz"));
        }
        return dir;
    }

    /** CIFAR-10's binary version (170 MB), downloaded and unpacked once; the directory to pass to {@code Cifar10.load}. */
    public static Path cifar10() throws IOException, InterruptedException {
        Path dir = dataDir().resolve("cifar10");
        if (!Files.isDirectory(dir.resolve("cifar-10-batches-bin"))) {
            untar(download("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", dir.resolve("cifar-10-binary.tar.gz")), dir);
        }
        return dir;
    }

    /** Unpacks a .tar.gz (or the listed members of it) into {@code dir} with the system's tar. */
    public static void untar(Path tgz, Path dir, String... members) throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>(List.of("tar", "xzf", tgz.toString(), "-C", dir.toString()));
        cmd.addAll(List.of(members));
        if (new ProcessBuilder(cmd).inheritIO().start().waitFor() != 0) {
            throw new IOException("could not unpack " + tgz);
        }
    }

    /** Unpacks a .zip into {@code dir}. */
    public static void unzip(Path zip, Path dir) throws IOException {
        try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
            for (ZipEntry e; (e = in.getNextEntry()) != null;) {
                Path out = dir.resolve(e.getName()).normalize();
                if (!out.startsWith(dir)) {
                    throw new IOException("bad entry " + e.getName());
                }
                if (e.isDirectory()) {
                    Files.createDirectories(out);
                } else {
                    Files.createDirectories(out.getParent());
                    Files.copy(in, out, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }

    /** Lines of a CSV file split on {@code separator}, without the header and without quotes. */
    public static List<String[]> csv(Path file, String separator) throws IOException {
        List<String[]> rows = new ArrayList<>();
        List<String> lines = Files.readAllLines(file);
        for (String line : lines.subList(1, lines.size())) {
            if (!line.isBlank()) {
                rows.add(line.replace("\"", "").split(separator));
            }
        }
        return rows;
    }

    /** Standardises the columns of [rows, cols] {@code x} with the mean and standard deviation of its first {@code fitRows} rows. */
    public static void standardise(float[] x, int rows, int cols, int fitRows) {
        for (int j = 0; j < cols; j++) {
            double mean = 0;
            double sq = 0;
            for (int i = 0; i < fitRows; i++) {
                mean += x[i * cols + j];
            }
            mean /= fitRows;
            for (int i = 0; i < fitRows; i++) {
                sq += (x[i * cols + j] - mean) * (x[i * cols + j] - mean);
            }
            double std = Math.max(1e-6, Math.sqrt(sq / fitRows));
            for (int i = 0; i < rows; i++) {
                x[i * cols + j] = (float) ((x[i * cols + j] - mean) / std);
            }
        }
    }

    /** A random permutation of 0 .. n-1. */
    public static int[] permutation(int n, long seed) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            order.add(i);
        }
        Collections.shuffle(order, new java.util.Random(seed));
        return order.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Tiny Shakespeare (1.1 MB of plays, from Andrej Karpathy's char-rnn), downloaded once. */
    public static Path shakespeare() throws IOException {
        return download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                dataDir().resolve("shakespeare").resolve("input.txt"));
    }

    /** The files of a Hugging Face model repository, downloaded once; the directory. */
    public static Path huggingFace(String repo, String... files) throws IOException {
        Path dir = dataDir().resolve("hf").resolve(repo);
        for (String f : files) {
            download("https://huggingface.co/" + repo + "/resolve/main/" + f, dir.resolve(f));
        }
        return dir;
    }

    /** Reads a gzipped or plain text resource of this module. */
    public static InputStream resource(String name) throws IOException {
        InputStream in = Examples.class.getResourceAsStream("/" + name);
        if (in == null) {
            throw new IOException("missing resource " + name);
        }
        return name.endsWith(".gz") ? new GZIPInputStream(in) : in;
    }

    /** The indices of the {@code k} largest values, largest first. */
    public static int[] topK(float[] values, int k) {
        Integer[] idx = new Integer[values.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Float.compare(values[b], values[a]));
        int[] out = new int[Math.min(k, idx.length)];
        for (int i = 0; i < out.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    /** Softmax of {@code logits[from, from + n)} at the given temperature. */
    public static float[] softmax(float[] logits, int from, int n, float temperature) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, logits[from + i]);
        }
        float[] p = new float[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            p[i] = (float) Math.exp((logits[from + i] - max) / temperature);
            sum += p[i];
        }
        for (int i = 0; i < n; i++) {
            p[i] /= (float) sum;
        }
        return p;
    }

    /** Draws an index from the probabilities {@code p}. */
    public static int sample(float[] p, java.util.Random random) {
        float u = random.nextFloat();
        float acc = 0;
        for (int i = 0; i < p.length; i++) {
            acc += p[i];
            if (u < acc) {
                return i;
            }
        }
        return p.length - 1;
    }

    /** Seconds since {@code t0} (from {@code System.nanoTime()}). */
    public static double seconds(long t0) {
        return (System.nanoTime() - t0) / 1e9;
    }

    /** The integer argument at {@code i}, or {@code fallback} when absent. */
    public static int intArg(String[] args, int i, int fallback) {
        return args.length > i ? Integer.parseInt(args[i]) : fallback;
    }
}
