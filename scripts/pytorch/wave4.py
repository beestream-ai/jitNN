"""PyTorch references for jitNN's wave 4 gates: a VAE, a DCGAN and a U-Net on MNIST, and LSTM and
GRU character language models on the text corpus. Built exactly like jitnn.nn.models (same layers,
parameter order and seeds, so bit-identical initial weights), trained on the same batches with
the same noise (jitNN's Philox, reproduced by jitnn_init.uniform).

usage: python wave4.py train  <vae|dcgan|unet|lstm|gru> <data dir> <out file>
       python wave4.py grad64 <vae|dcgan|unet|lstm|gru> <data dir> <out prefix>

<data dir> is the MNIST directory (vae, dcgan, unet) or the directory holding corpus.txt (lstm,
gru). train writes float32 [evaluation metrics..., loss per step (two per step for dcgan: D, G)];
grad64 writes the step-0 gradients in FP64 (<prefix>.grad0_f64) and PyTorch's FP32 relative error
per parameter (<prefix>.grad0_f32err); for dcgan one pair per network (<prefix>_d, <prefix>_g).
"""
import gzip
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True

from jitnn_init import conv2d, linear, uniform

# Shared with GenerativeAccelTest and RecurrentAccelTest.
VAE = dict(batch=128, steps=400, hidden=400, latent=20, lr=1e-3, noise=5000, eval_noise=9000, eval_batches=10)
GAN = dict(batch=64, steps=300, latent=64, width=32, lr=2e-4, beta1=0.5, noise=6000, g_seed=1, d_seed=50)
UNET = dict(batch=32, steps=300, width=16, lr=1e-3, eval_batches=32)
RNN = dict(batch=32, seq=64, dim=64, hidden=256, layers=2, lr=2e-3, clip=1.0, steps=300)
SQRT3 = np.float32(math.sqrt(3.0))


class Seeds:
    def __init__(self, base):
        self.next_seed = base

    def __call__(self):
        s = self.next_seed
        self.next_seed += 1
        return s


def t(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def fc(i, o, seed):
    m = torch.nn.Linear(i, o)
    w, b = linear(i, o, seed)
    with torch.no_grad():
        m.weight.copy_(t(w))
        m.bias.copy_(t(b))
    return m


def conv(i, o, k, stride, pad, seed, bias=True):
    m = torch.nn.Conv2d(i, o, k, stride=stride, padding=pad, bias=bias)
    w, b = conv2d(i, o, k, seed, bias=bias)
    with torch.no_grad():
        m.weight.copy_(t(w))
        if bias:
            m.bias.copy_(t(b))
    return m


def conv_t(i, o, k, stride, pad, seed, bias=True):
    """As jitnn.nn.ConvTranspose2d: weight [in, out, k, k], bound 1/sqrt(out k k)."""
    m = torch.nn.ConvTranspose2d(i, o, k, stride=stride, padding=pad, bias=bias)
    bound = np.float32(1.0 / np.sqrt(o * k * k))
    with torch.no_grad():
        m.weight.copy_(t(uniform(seed, -bound, bound, i * o * k * k).reshape(i, o, k, k)))
        if bias:
            m.bias.copy_(t(uniform(seed + 0x9E3779B97F4A7C15, -bound, bound, o)))
    return m


def noise(seed, *dims):
    """jitNN's Tensor.uniform(seed, -sqrt(3), sqrt(3), dims): unit-variance uniform noise."""
    return t(uniform(seed, -SQRT3, SQRT3, int(np.prod(dims))).reshape(dims))


# ---- models

class Vae(torch.nn.Module):
    def __init__(self, n_in, hidden, latent, seed):
        super().__init__()
        s = Seeds(seed)
        self.enc, self.mu, self.logvar = fc(n_in, hidden, s()), fc(hidden, latent, s()), fc(hidden, latent, s())
        self.dec, self.out = fc(latent, hidden, s()), fc(hidden, n_in, s())

    def forward(self, x, eps):
        h = torch.relu(self.enc(x))
        m, lv = self.mu(h), self.logvar(h)
        z = m + (lv * 0.5).exp() * eps
        return torch.sigmoid(self.out(torch.relu(self.dec(z)))), m, lv


def vae_loss(recon, m, lv, x):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kl = (1 + lv - m.square() - lv.exp()).sum() * -0.5
    return (bce + kl) * (1.0 / x.shape[0])


class Generator(torch.nn.Module):
    def __init__(self, latent, width, seed):
        super().__init__()
        s = Seeds(seed)
        self.width = width
        self.fc = fc(latent, 2 * width * 49, s())
        self.bn0 = torch.nn.BatchNorm2d(2 * width)
        self.up1 = conv_t(2 * width, width, 4, 2, 1, s(), bias=False)
        self.bn1 = torch.nn.BatchNorm2d(width)
        self.up2 = conv_t(width, 1, 4, 2, 1, s())

    def forward(self, z):
        x = torch.relu(self.bn0(self.fc(z).reshape(z.shape[0], 2 * self.width, 7, 7)))
        x = torch.relu(self.bn1(self.up1(x)))
        return torch.tanh(self.up2(x))


class Discriminator(torch.nn.Module):
    def __init__(self, width, seed):
        super().__init__()
        s = Seeds(seed)
        self.width = width
        self.c1 = conv(1, width, 4, 2, 1, s())
        self.c2 = conv(width, 2 * width, 4, 2, 1, s(), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(2 * width)
        self.fc = fc(2 * width * 49, 1, s())

    def forward(self, x):
        h = F.leaky_relu(self.c1(x), 0.2)
        h = F.leaky_relu(self.bn2(self.c2(h)), 0.2)
        return self.fc(h.reshape(x.shape[0], -1))


class UNet(torch.nn.Module):
    def __init__(self, n_in, w, seed):
        super().__init__()
        s = Seeds(seed)
        c = lambda i, o, k: conv(i, o, k, 1, k // 2, s())
        self.e1a, self.e1b = c(n_in, w, 3), c(w, w, 3)
        self.e2a, self.e2b = c(w, 2 * w, 3), c(2 * w, 2 * w, 3)
        self.ba, self.bb = c(2 * w, 4 * w, 3), c(4 * w, 4 * w, 3)
        self.up2 = conv_t(4 * w, 2 * w, 2, 2, 0, s())
        self.d2a, self.d2b = c(4 * w, 2 * w, 3), c(2 * w, 2 * w, 3)
        self.up1 = c(2 * w, w, 1)
        self.d1a, self.d1b = c(2 * w, w, 3), c(w, w, 3)
        self.head = c(w, 1, 1)

    def forward(self, x):
        r = torch.relu
        f1 = r(self.e1b(r(self.e1a(x))))
        f2 = r(self.e2b(r(self.e2a(F.max_pool2d(f1, 2, 2)))))
        b = r(self.bb(r(self.ba(F.max_pool2d(f2, 2, 2)))))
        u2 = r(self.d2b(r(self.d2a(torch.cat([self.up2(b), f2], 1)))))
        u1 = self.up1(F.interpolate(u2, scale_factor=2.0, mode="bilinear", align_corners=False))
        return self.head(r(self.d1b(r(self.d1a(torch.cat([u1, f1], 1))))))


class CharRnn(torch.nn.Module):
    def __init__(self, vocab, dim, hidden, layers, gru, seed):
        super().__init__()
        self.emb = torch.nn.Parameter(t(uniform(seed, np.float32(-0.1), np.float32(0.1), vocab * dim).reshape(vocab, dim)))
        cls = torch.nn.GRU if gru else torch.nn.LSTM
        gates = 3 if gru else 4
        self.rnn = cls(dim, hidden, layers)
        bound = np.float32(1.0 / np.sqrt(hidden))
        with torch.no_grad():
            for l in range(layers):
                n_in = dim if l == 0 else hidden
                base = seed + 1 + 4 * l
                getattr(self.rnn, f"weight_ih_l{l}").copy_(t(uniform(base, -bound, bound, gates * hidden * n_in).reshape(gates * hidden, n_in)))
                getattr(self.rnn, f"weight_hh_l{l}").copy_(t(uniform(base + 1, -bound, bound, gates * hidden * hidden).reshape(gates * hidden, hidden)))
                getattr(self.rnn, f"bias_ih_l{l}").copy_(t(uniform(base + 2, -bound, bound, gates * hidden)))
                getattr(self.rnn, f"bias_hh_l{l}").copy_(t(uniform(base + 3, -bound, bound, gates * hidden)))
        self.head = fc(hidden, vocab, seed + 100)

    def forward(self, tokens):
        y, _ = self.rnn(self.emb[tokens])
        return self.head(y)

    def ordered_parameters(self):
        """jitNN's order: embedding, per layer W_ih, W_hh, b_ih, b_hh, then the head."""
        ps = [self.emb]
        for l in range(self.rnn.num_layers):
            ps += [getattr(self.rnn, f"{n}_l{l}") for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
        return ps + [self.head.weight, self.head.bias]


# ---- data, laid out exactly as jitnn.data

def mnist(d, train):
    prefix = "train" if train else "t10k"

    def read(name, offset):
        path = os.path.join(d, name)
        raw = open(path, "rb").read() if os.path.exists(path) else gzip.open(path + ".gz").read()
        return np.frombuffer(raw, dtype=np.uint8, offset=offset)

    raw = read(f"{prefix}-images-idx3-ubyte", 16).reshape(-1, 1, 28, 28)
    return raw, read(f"{prefix}-labels-idx1-ubyte", 8)


def unit(raw):
    return raw.astype(np.float32) / np.float32(255.0)


def mnist_batch(raw, b, size):
    n = raw.shape[0] // size
    return raw[(b % n) * size:(b % n + 1) * size]


def corpus(d):
    tokens = np.frombuffer(open(os.path.join(d, "corpus.txt"), "rb").read(), dtype=np.uint8).astype(np.int64)
    return tokens, len(tokens) * 9 // 10


def lm_batch(tokens, split, b):
    rows, seq = RNN["batch"], RNN["seq"]
    starts = [((b * rows + r) * seq) % (split - seq - 1) for r in range(rows)]
    x = np.stack([tokens[s:s + seq] for s in starts]).T  # [T, B]
    y = np.stack([tokens[s + 1:s + seq + 1] for s in starts]).T
    return t(x), t(y)


def device():
    return torch.device(os.environ.get("JITNN_TORCH_DEVICE", "cpu"))


# ---- one training step per gate; each returns the losses and leaves the gradients in place

class Gate:
    def __init__(self, kind, d, dtype, dev):
        self.kind, self.dtype, self.dev = kind, dtype, dev
        if kind == "vae":
            self.raw, _ = mnist(d, True)
            self.test, _ = mnist(d, False)
            self.net = Vae(784, VAE["hidden"], VAE["latent"], 1).to(dtype).to(dev)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=VAE["lr"])
        elif kind == "dcgan":
            self.raw, _ = mnist(d, True)
            self.g = Generator(GAN["latent"], GAN["width"], GAN["g_seed"]).to(dtype).to(dev)
            self.d = Discriminator(GAN["width"], GAN["d_seed"]).to(dtype).to(dev)
            betas = (GAN["beta1"], 0.999)
            self.g_opt = torch.optim.Adam(self.g.parameters(), lr=GAN["lr"], betas=betas)
            self.d_opt = torch.optim.Adam(self.d.parameters(), lr=GAN["lr"], betas=betas)
        elif kind == "unet":
            self.raw, _ = mnist(d, True)
            self.test, _ = mnist(d, False)
            self.net = UNet(1, UNET["width"], 1).to(dtype).to(dev)
            self.opt = torch.optim.Adam(self.net.parameters(), lr=UNET["lr"])
        else:
            self.tokens, self.split = corpus(d)
            self.net = CharRnn(256, RNN["dim"], RNN["hidden"], RNN["layers"], kind == "gru", 1).to(dtype).to(dev)
            self.opt = torch.optim.Adam(self.net.ordered_parameters(), lr=RNN["lr"])

    def x(self, a):
        return t(a).to(self.dtype).to(self.dev)

    def params(self):
        if self.kind == "dcgan":
            return list(self.d.parameters()), list(self.g.parameters())
        if self.kind in ("lstm", "gru"):
            return (self.net.ordered_parameters(),)
        return (list(self.net.parameters()),)

    def step(self, b, update=True, on_d_grads=None):
        if self.kind == "vae":
            x = self.x(unit(mnist_batch(self.raw, b, VAE["batch"])).reshape(-1, 784))
            loss = vae_loss(*self.net(x, noise(VAE["noise"] + b, VAE["batch"], VAE["latent"]).to(self.dtype).to(self.dev)), x)
            self.opt.zero_grad()
            loss.backward()
            if update:
                self.opt.step()
            return [loss.item()]
        if self.kind == "dcgan":
            n = GAN["batch"]
            real = self.x(unit(mnist_batch(self.raw, b, n))) * 2 - 1
            z = noise(GAN["noise"] + b, n, GAN["latent"]).to(self.dtype).to(self.dev)
            ones = torch.ones(n, 1, dtype=self.dtype, device=self.dev)
            zeros = torch.zeros(n, 1, dtype=self.dtype, device=self.dev)
            fake = self.g(z)
            d_loss = F.binary_cross_entropy_with_logits(self.d(real), ones) + F.binary_cross_entropy_with_logits(self.d(fake.detach()), zeros)
            self.d_opt.zero_grad()
            d_loss.backward()
            if on_d_grads is not None:
                on_d_grads()
            self.d_opt.step()
            g_loss = F.binary_cross_entropy_with_logits(self.d(fake), ones)
            self.g_opt.zero_grad()
            g_loss.backward()
            if update:
                self.g_opt.step()
            return [d_loss.item(), g_loss.item()]
        if self.kind == "unet":
            raw = mnist_batch(self.raw, b, UNET["batch"])
            loss = F.binary_cross_entropy_with_logits(self.net(self.x(unit(raw))), self.x((raw > 127).astype(np.float32)))
            self.opt.zero_grad()
            loss.backward()
            if update:
                self.opt.step()
            return [loss.item()]
        x, y = lm_batch(self.tokens, self.split, b)
        logits = self.net(x.to(self.dev))
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1).to(self.dev))
        self.opt.zero_grad()
        loss.backward()
        if update:
            torch.nn.utils.clip_grad_norm_(self.net.ordered_parameters(), RNN["clip"])
            self.opt.step()
        return [loss.item()]

    def evaluate(self):
        """Metrics after training, as the Java gate computes them."""
        with torch.no_grad():
            if self.kind == "vae":
                self.net.eval()
                total = 0.0
                for i in range(VAE["eval_batches"]):
                    x = self.x(unit(mnist_batch(self.test, i, VAE["batch"])).reshape(-1, 784))
                    eps = noise(VAE["eval_noise"] + i, VAE["batch"], VAE["latent"]).to(self.dtype).to(self.dev)
                    total += vae_loss(*self.net(x, eps), x).item()
                return [total / VAE["eval_batches"]]
            if self.kind == "dcgan":
                return []
            if self.kind == "unet":
                self.net.eval()
                loss, inter, union = 0.0, 0.0, 0.0
                for i in range(UNET["eval_batches"]):
                    raw = mnist_batch(self.test, i, UNET["batch"])
                    target = self.x((raw > 127).astype(np.float32))
                    logits = self.net(self.x(unit(raw)))
                    loss += F.binary_cross_entropy_with_logits(logits, target).item()
                    pred = (logits > 0).to(self.dtype)
                    inter += (pred * target).sum().item()
                    union += ((pred + target) > 0).to(self.dtype).sum().item()
                return [loss / UNET["eval_batches"], inter / union]
            seq, rows = RNN["seq"], RNN["batch"]
            windows = (len(self.tokens) - self.split - 1) // seq
            total, count = 0.0, 0
            for w0 in range(0, windows - rows + 1, rows):
                starts = [self.split + (w0 + r) * seq for r in range(rows)]
                x = t(np.stack([self.tokens[s:s + seq] for s in starts]).T)
                y = t(np.stack([self.tokens[s + 1:s + seq + 1] for s in starts]).T)
                logits = self.net(x.to(self.dev))
                total += F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1).to(self.dev)).item()
                count += 1
            return [total / count]


def steps_of(kind):
    return {"vae": VAE, "dcgan": GAN, "unet": UNET}.get(kind, RNN)["steps"]


def train(kind, d, out):
    torch.set_num_threads(8)
    gate = Gate(kind, d, torch.float32, device())
    losses = []
    for b in range(steps_of(kind)):
        losses += gate.step(b)
    metrics = gate.evaluate()
    np.array(metrics + losses, dtype=np.float32).tofile(out)
    print(f"{kind}: {steps_of(kind)} steps, first losses {losses[:2]}, last {np.mean(losses[-50:]):.4f}, metrics {metrics}")


def grad64(kind, d, prefix):
    grads = {}
    for dtype in (torch.float64, torch.float32):
        gate = Gate(kind, d, dtype, device())
        if kind == "dcgan":
            captured = {}
            gate.step(0, update=False, on_d_grads=lambda: captured.update(d=[p.grad.detach().double().cpu().numpy().ravel().copy() for p in gate.d.parameters()]))
            grads[dtype] = [captured["d"], [p.grad.detach().double().cpu().numpy().ravel() for p in gate.g.parameters()]]
        else:
            gate.step(0, update=False)
            grads[dtype] = [[p.grad.detach().double().cpu().numpy().ravel() for p in gate.params()[0]]]
    names = [prefix + "_d", prefix + "_g"] if kind == "dcgan" else [prefix]
    for i, name in enumerate(names):
        g64, g32 = grads[torch.float64][i], grads[torch.float32][i]
        err = [np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30) for a, b in zip(g64, g32)]
        np.concatenate(g64).astype(np.float64).tofile(name + ".grad0_f64")
        np.array(err, dtype=np.float32).tofile(name + ".grad0_f32err")
        print(f"{name}: {len(err)} parameters, PyTorch FP32 vs FP64 relative error: max {max(err):.2e}, median {np.median(err):.2e}")


if __name__ == "__main__":
    mode, kind, d, out = sys.argv[1:5]
    (train if mode == "train" else grad64)(kind, d, out)
