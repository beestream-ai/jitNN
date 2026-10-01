package org.beehive.jitnn.nn.models;

import java.util.ArrayList;
import java.util.List;

import org.beehive.jitnn.nn.Add;
import org.beehive.jitnn.nn.BatchNorm2d;
import org.beehive.jitnn.nn.Conv2d;
import org.beehive.jitnn.nn.GlobalAvgPool2d;
import org.beehive.jitnn.nn.Linear;
import org.beehive.jitnn.nn.MaxPool2d;
import org.beehive.jitnn.nn.Module;
import org.beehive.jitnn.nn.ReLU;
import org.beehive.jitnn.nn.Sequential;

/**
 * ResNet (He et al., 2016). {@link #resnet18} is torchvision's resnet18 layout; the {@code small}
 * variant replaces the 7x7/2 stem and max pool with a 3x3/1 stem for 28-32 pixel inputs (the usual
 * CIFAR adaptation). Convolutions have no bias, every one is followed by BatchNorm.
 */
public final class ResNet {

    private ResNet() {
    }

    /** ResNet-18: BasicBlocks [2, 2, 2, 2], widths 64-512. */
    public static Sequential resnet18(int inChannels, int classes, boolean small, long seed) {
        return build(new int[] { 2, 2, 2, 2 }, inChannels, classes, small, new Seeds(seed));
    }

    private static Sequential build(int[] blocks, int inChannels, int classes, boolean small, Seeds seeds) {
        List<Module> layers = new ArrayList<>();
        if (small) {
            layers.add(new Conv2d(inChannels, 64, 3, 1, 1, 1, 1, false, seeds.next()));
            layers.add(new BatchNorm2d(64));
            layers.add(new ReLU());
        } else {
            layers.add(new Conv2d(inChannels, 64, 7, 2, 3, 1, 1, false, seeds.next()));
            layers.add(new BatchNorm2d(64));
            layers.add(new ReLU());
            layers.add(new MaxPool2d(3, 2, 1));
        }
        int in = 64;
        int[] widths = { 64, 128, 256, 512 };
        for (int stage = 0; stage < 4; stage++) {
            for (int b = 0; b < blocks[stage]; b++) {
                int stride = stage > 0 && b == 0 ? 2 : 1;
                layers.add(basicBlock(in, widths[stage], stride, seeds));
                in = widths[stage];
            }
        }
        layers.add(new GlobalAvgPool2d());
        layers.add(new Linear(512, classes, seeds.next()));
        return new Sequential(layers.toArray(new Module[0]));
    }

    /** conv3x3-bn-relu-conv3x3-bn, plus a 1x1-conv-bn shortcut when the shape changes; then add and ReLU. */
    static Module basicBlock(int in, int out, int stride, Seeds seeds) {
        Sequential main = new Sequential(
                new Conv2d(in, out, 3, stride, 1, 1, 1, false, seeds.next()), new BatchNorm2d(out), new ReLU(),
                new Conv2d(out, out, 3, 1, 1, 1, 1, false, seeds.next()), new BatchNorm2d(out));
        Sequential shortcut = stride != 1 || in != out
                ? new Sequential(new Conv2d(in, out, 1, stride, 0, 1, 1, false, seeds.next()), new BatchNorm2d(out))
                : null;
        return new Add(main, shortcut, true);
    }
}
