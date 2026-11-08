package ai.beestream.jitnn.nn.models;

import java.util.ArrayList;
import java.util.List;

import ai.beestream.jitnn.nn.Add;
import ai.beestream.jitnn.nn.BatchNorm2d;
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.GlobalAvgPool2d;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.MaxPool2d;
import ai.beestream.jitnn.nn.Module;
import ai.beestream.jitnn.nn.ReLU;
import ai.beestream.jitnn.nn.Sequential;

/**
 * ResNet (He et al., 2016). {@link #resnet18} and {@link #resnet50} are torchvision's layouts; the {@code small}
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

    /** ResNet-50: Bottleneck blocks [3, 4, 6, 3], widths 64-512, expansion 4 (torchvision's v1.5, stride on the 3x3). */
    public static Sequential resnet50(int inChannels, int classes, boolean small, long seed) {
        return build(new int[] { 3, 4, 6, 3 }, true, inChannels, classes, small, new Seeds(seed));
    }

    private static Sequential build(int[] blocks, int inChannels, int classes, boolean small, Seeds seeds) {
        return build(blocks, false, inChannels, classes, small, seeds);
    }

    private static Sequential build(int[] blocks, boolean bottleneck, int inChannels, int classes, boolean small, Seeds seeds) {
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
                if (bottleneck) {
                    layers.add(bottleneckBlock(in, widths[stage], stride, seeds));
                    in = 4 * widths[stage];
                } else {
                    layers.add(basicBlock(in, widths[stage], stride, seeds));
                    in = widths[stage];
                }
            }
        }
        layers.add(new GlobalAvgPool2d());
        layers.add(new Linear(in, classes, seeds.next()));
        return new Sequential(layers.toArray(new Module[0]));
    }

    /**
     * conv1x1-bn-relu-conv3x3(stride)-bn-relu-conv1x1(4 width)-bn, plus a 1x1-conv-bn shortcut when
     * the shape changes; then add and ReLU.
     */
    static Module bottleneckBlock(int in, int width, int stride, Seeds seeds) {
        int out = 4 * width;
        Sequential main = new Sequential(
                new Conv2d(in, width, 1, 1, 0, 1, 1, false, seeds.next()), new BatchNorm2d(width), new ReLU(),
                new Conv2d(width, width, 3, stride, 1, 1, 1, false, seeds.next()), new BatchNorm2d(width), new ReLU(),
                new Conv2d(width, out, 1, 1, 0, 1, 1, false, seeds.next()), new BatchNorm2d(out));
        Sequential shortcut = stride != 1 || in != out
                ? new Sequential(new Conv2d(in, out, 1, stride, 0, 1, 1, false, seeds.next()), new BatchNorm2d(out))
                : null;
        return new Add(main, shortcut, true);
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
