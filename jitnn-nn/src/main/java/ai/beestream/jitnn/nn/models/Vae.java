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

import ai.beestream.jitnn.F;
import ai.beestream.jitnn.Tensor;
import ai.beestream.jitnn.nn.Linear;
import ai.beestream.jitnn.nn.Module;

/**
 * A fully connected variational autoencoder (Kingma and Welling, 2014), the layout of PyTorch's
 * MNIST example: 784-400 encoder to a Gaussian latent (mean, log-variance), 400-784 decoder with
 * a sigmoid output. Seeds, in order: encoder, mean, log-variance, decoder hidden, decoder output.
 */
public final class Vae extends Module {

    /** Reconstruction probabilities [B, in] and the latent mean and log-variance [B, latent]. */
    public record Output(Tensor recon, Tensor mu, Tensor logvar) {
    }

    private final Linear enc;
    private final Linear mu;
    private final Linear logvar;
    private final Linear dec;
    private final Linear out;
    private final int latent;

    /**
     * An autoencoder of {@code in} features through {@code hidden} to a latent of {@code latent} values,
     * initialised from {@code seed}.
     */
    public Vae(int in, int hidden, int latent, long seed) {
        Seeds seeds = new Seeds(seed);
        this.latent = latent;
        this.enc = child(new Linear(in, hidden, seeds.next()));
        this.mu = child(new Linear(hidden, latent, seeds.next()));
        this.logvar = child(new Linear(hidden, latent, seeds.next()));
        this.dec = child(new Linear(latent, hidden, seeds.next()));
        this.out = child(new Linear(hidden, in, seeds.next()));
    }

    /** The size of the latent vector. */
    public int latent() {
        return latent;
    }

    /** Encodes x [B, in], samples z = mu + exp(logvar / 2) eps with the given noise eps [B, latent], decodes. */
    public Output run(Tensor x, Tensor eps) {
        Tensor h = enc.forwardRelu(x);
        Tensor m = mu.forward(h);
        Tensor lv = logvar.forward(h);
        Tensor z = m.add(lv.mul(0.5f).exp().mul(eps));
        return new Output(decode(z), m, lv);
    }

    /** Decodes latent vectors z [B, latent] to [B, in], each value in (0, 1). */
    public Tensor decode(Tensor z) {
        return out.forward(dec.forwardRelu(z)).sigmoid();
    }

    /** The negative ELBO per sample: (binary cross-entropy summed over pixels + KL) / B. */
    public static Tensor loss(Output o, Tensor x) {
        Tensor bce = F.binaryCrossEntropy(o.recon(), x, F.Reduction.SUM);
        return bce.add(F.gaussianKl(o.mu(), o.logvar())).mul(1.0f / x.dim(0));
    }

    /** Reconstruction with the mean latent (no sampling). */
    @Override
    public Tensor forward(Tensor x) {
        return decode(mu.forward(enc.forwardRelu(x)));
    }
}
