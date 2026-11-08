package ai.beestream.jitnn.nn.models;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.nn.Add;
import ai.beestream.jitnn.nn.BatchNorm2d;
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.Dropout;
import ai.beestream.jitnn.nn.GlobalAvgPool2d;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU6;
import ai.beestream.jitnn.nn.Sequential;

/**
 * MobileNetV2 (Sandler et al., 2018), torchvision's layout: inverted residual blocks (1x1 expand,
 * 3x3 depthwise, 1x1 linear projection) with ReLU6. The {@code small} variant uses stride 1 in the
 * stem and the first stride-2 stage, for 28-32 pixel inputs.
 */
public final class MobileNetV2 {

    private MobileNetV2() {
    }

    /** t (expansion), c (channels), n (repeats), s (first stride), as in the paper's table 2. */
    private static final int[][] SETTINGS = { { 1, 16, 1, 1 }, { 6, 24, 2, 2 }, { 6, 32, 3, 2 }, { 6, 64, 4, 2 }, { 6, 96, 3, 1 }, { 6, 160, 3, 2 }, { 6, 320, 1, 1 } };

    /**
     * MobileNetV2 over images of {@code inChannels} channels with {@code classes} outputs and dropout
     * {@code dropout} before the classifier, initialised from {@code seed}.
     */
    public static Sequential mobilenetV2(int inChannels, int classes, boolean small, float dropout, long seed) {
        Seeds seeds = new Seeds(seed);
        List<Module> layers = new ArrayList<>();
        int in = 32;
        layers.add(new Conv2d(inChannels, in, 3, small ? 1 : 2, 1, 1, 1, false, seeds.next()));
        layers.add(new BatchNorm2d(in));
        layers.add(new ReLU6());
        for (int i = 0; i < SETTINGS.length; i++) {
            int[] st = SETTINGS[i];
            for (int r = 0; r < st[2]; r++) {
                int stride = r == 0 ? st[3] : 1;
                if (small && i == 1) {
                    stride = 1;
                }
                layers.add(invertedResidual(in, st[1], stride, st[0], seeds));
                in = st[1];
            }
        }
        layers.add(new Conv2d(in, 1280, 1, 1, 0, 1, 1, false, seeds.next()));
        layers.add(new BatchNorm2d(1280));
        layers.add(new ReLU6());
        layers.add(new GlobalAvgPool2d());
        if (dropout > 0) {
            layers.add(new Dropout(dropout, seeds.next()));
        }
        layers.add(new Linear(1280, classes, seeds.next()));
        return new Sequential(layers.toArray(new Module[0]));
    }

    static Module invertedResidual(int in, int out, int stride, int expand, Seeds seeds) {
        int hidden = in * expand;
        List<Module> m = new ArrayList<>();
        if (expand != 1) {
            m.add(new Conv2d(in, hidden, 1, 1, 0, 1, 1, false, seeds.next()));
            m.add(new BatchNorm2d(hidden));
            m.add(new ReLU6());
        }
        m.add(new Conv2d(hidden, hidden, 3, stride, 1, 1, hidden, false, seeds.next()));
        m.add(new BatchNorm2d(hidden));
        m.add(new ReLU6());
        m.add(new Conv2d(hidden, out, 1, 1, 0, 1, 1, false, seeds.next()));
        m.add(new BatchNorm2d(out));
        Sequential main = new Sequential(m.toArray(new Module[0]));
        return stride == 1 && in == out ? new Add(main, null, false) : main;
    }
}
