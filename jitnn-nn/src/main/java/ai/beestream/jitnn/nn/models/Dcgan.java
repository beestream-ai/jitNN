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
package ai.beestream.jitnn.nn.models;

import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.BatchNorm2d;
import ai.beestream.jitnn.nn.Conv2d;
import ai.beestream.jitnn.nn.ConvTranspose2d;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;

/**
 * DCGAN (Radford et al., 2016) for 28x28 single-channel images: a generator from a latent vector
 * through a 7x7 feature map and two stride-2 transposed convolutions to a tanh image, and a
 * discriminator of two stride-2 convolutions with LeakyReLU(0.2) and a linear logit. BatchNorm
 * everywhere but the generator's output and the discriminator's input layer.
 */
public final class Dcgan {

    private Dcgan() {
    }

    /** z [B, latent] to images [B, 1, 28, 28] in [-1, 1]. Seeds: linear, then the two transposed convolutions. */
    public static final class Generator extends Module {

        private final Linear fc;
        private final BatchNorm2d bn0;
        private final ConvTranspose2d up1;
        private final BatchNorm2d bn1;
        private final ConvTranspose2d up2;
        private final int width;

        /**
         * A generator from latent vectors of {@code latent} values, {@code width} channels in its last
         * hidden layer, initialised from {@code seed}.
         */
        public Generator(int latent, int width, long seed) {
            Seeds seeds = new Seeds(seed);
            this.width = width;
            this.fc = child(new Linear(latent, 2 * width * 7 * 7, seeds.next()));
            this.bn0 = child(new BatchNorm2d(2 * width));
            this.up1 = child(new ConvTranspose2d(2 * width, width, 4, 2, 1, 0, 1, 1, false, seeds.next()));
            this.bn1 = child(new BatchNorm2d(width));
            this.up2 = child(new ConvTranspose2d(width, 1, 4, 2, 1, 0, 1, 1, true, seeds.next()));
        }

        @Override
        public Tensor forward(Tensor z) {
            Tensor x = bn0.forwardRelu(fc.forward(z).reshape(z.dim(0), 2 * width, 7, 7));
            x = bn1.forwardRelu(up1.forward(x));
            return up2.forward(x).tanh();
        }
    }

    /** Images [B, 1, 28, 28] to logits [B, 1]. Seeds: the two convolutions, then the linear layer. */
    public static final class Discriminator extends Module {

        private final Conv2d c1;
        private final Conv2d c2;
        private final BatchNorm2d bn2;
        private final Linear fc;
        private final int width;

        /** A discriminator with {@code width} channels in its first layer, initialised from {@code seed}. */
        public Discriminator(int width, long seed) {
            Seeds seeds = new Seeds(seed);
            this.width = width;
            this.c1 = child(new Conv2d(1, width, 4, 2, 1, 1, 1, true, seeds.next()));
            this.c2 = child(new Conv2d(width, 2 * width, 4, 2, 1, 1, 1, false, seeds.next()));
            this.bn2 = child(new BatchNorm2d(2 * width));
            this.fc = child(new Linear(2 * width * 7 * 7, 1, seeds.next()));
        }

        @Override
        public Tensor forward(Tensor x) {
            Tensor h = c1.forward(x).leakyRelu(0.2f);
            h = bn2.forward(c2.forward(h)).leakyRelu(0.2f);
            return fc.forward(h.reshape(x.dim(0), 2 * width * 7 * 7));
        }
    }
}
