package org.beehive.jitnn.examples;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.beehive.jitnn.Tensor;
import org.beehive.jitnn.data.Mnist;
import org.beehive.jitnn.ml.Tsne;

/**
 * A map of handwritten digits, after Smile's t-SNE demo: exact t-SNE on the device embeds 2,000
 * MNIST images (784 pixels each) in two dimensions, drawn into tsne-mnist.png with one colour per
 * digit. How often a point's nearest neighbours in the map share its digit shows how well the map
 * separates them.
 *
 * <pre>run.sh SmileTsneMnist [images]</pre>
 */
public final class SmileTsneMnist {

    private static final Color[] COLOURS = { new Color(0x1f77b4), new Color(0xff7f0e), new Color(0x2ca02c), new Color(0xd62728), new Color(0x9467bd),
            new Color(0x8c564b), new Color(0xe377c2), new Color(0x7f7f7f), new Color(0xbcbd22), new Color(0x17becf) };

    public static void main(String[] args) throws Exception {
        int n = Examples.intArg(args, 0, 2000);
        Mnist test = Mnist.load(Examples.mnist(), false, true);
        float[] x = Arrays.copyOf(test.images, n * Mnist.PIXELS);
        int[] digit = Arrays.copyOf(test.labels, n);

        long t0 = System.nanoTime();
        float[] map = new Tsne(2, 30, 1000).fitTransform(Tensor.of(x, n, Mnist.PIXELS)).toFloatArray();
        System.out.printf("t-SNE of %,d images: %.1f s%n", n, Examples.seconds(t0));
        System.out.printf("5 nearest neighbours in the map with the same digit: %.1f%%%n", 100 * neighbourAgreement(map, digit, 5));

        int size = 900;
        float minX = Float.MAX_VALUE, maxX = -Float.MAX_VALUE, minY = Float.MAX_VALUE, maxY = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            minX = Math.min(minX, map[2 * i]);
            maxX = Math.max(maxX, map[2 * i]);
            minY = Math.min(minY, map[2 * i + 1]);
            maxY = Math.max(maxY, map[2 * i + 1]);
        }
        BufferedImage img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, size, size);
        g.setStroke(new BasicStroke(1));
        for (int i = 0; i < n; i++) {
            int px = (int) (30 + (map[2 * i] - minX) / (maxX - minX) * (size - 60));
            int py = (int) (30 + (map[2 * i + 1] - minY) / (maxY - minY) * (size - 60));
            g.setColor(COLOURS[digit[i]]);
            g.fillOval(px - 3, py - 3, 6, 6);
        }
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 16));
        for (int k = 0; k < 10; k++) {
            g.setColor(COLOURS[k]);
            g.drawString(Integer.toString(k), size - 30, 30 + 22 * k);
        }
        g.dispose();
        Path png = Path.of("tsne-mnist.png");
        ImageIO.write(img, "png", png.toFile());
        System.out.println("wrote the map to " + png.toAbsolutePath());
    }

    /** The share of each point's k nearest neighbours in the 2-D map that have its label. */
    static float neighbourAgreement(float[] map, int[] label, int k) {
        int n = label.length;
        long same = 0;
        float[] dist = new float[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float dx = map[2 * i] - map[2 * j];
                float dy = map[2 * i + 1] - map[2 * j + 1];
                dist[j] = j == i ? -Float.MAX_VALUE : -(dx * dx + dy * dy);   // nearest = largest
            }
            for (int j : Examples.topK(dist, k)) {
                same += label[j] == label[i] ? 1 : 0;
            }
        }
        return (float) same / (n * k);
    }
}
