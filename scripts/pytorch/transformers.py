# Copyright (c) 2026 Beestream Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch references for jitNN's transformer gates: GPT-2, a Llama-style decoder and BERT,
built exactly like jitnn.nn.models (same layers, parameter order and seeds), trained on the same
byte-level corpus batches with AdamW, linear warm-up and cosine decay, and gradient clipping.

usage: python transformers.py train  <gpt2|llama|bert> <corpus file> <out file> <steps>
       python transformers.py grad64 <gpt2|llama|bert> <corpus file> <out prefix>

train writes [eval loss, loss per step] as float32 and the step-0 gradients (before clipping)
to <out>.grad0; grad64 writes the step-0 gradients in FP64 to <prefix>.grad0_f64 and PyTorch's
FP32 relative error per parameter to <prefix>.grad0_f32err.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from jitnn_init import linear, uniform

# Shared by TransformerAccelTest.
BATCH, SEQ, DIM, HEADS, LAYERS, LLAMA_HIDDEN = 16, 128, 128, 4, 4, 352
VOCAB, MASK, PAD, BERT_VOCAB, PREDICTIONS = 256, 256, 257, 258, 19
LR, WEIGHT_DECAY, WARMUP, MIN_RATIO, CLIP = 1e-3, 0.01, 30, 0.1, 1.0


class Seeds:
    def __init__(self, base):
        self.next_seed = base

    def __call__(self):
        s = self.next_seed
        self.next_seed += 1
        return s


def uniform_std(seed, std, *dims):
    a = np.float32(std * math.sqrt(3.0))
    return torch.from_numpy(uniform(seed, -a, a, int(np.prod(dims))).reshape(dims))


def fc(i, o, seed, bias=True):
    m = torch.nn.Linear(i, o, bias=bias)
    w, b = linear(i, o, seed)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(w))
        if bias:
            m.bias.copy_(torch.from_numpy(b))
    return m


def mha(x, qkv, proj, heads, mask, causal):
    b, t, d = x.shape
    q, k, v = qkv(x).reshape(b, t, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
    return proj(y.permute(0, 2, 1, 3).reshape(b, t, d))


class GPT2Block(torch.nn.Module):
    def __init__(self, dim, heads, seeds):
        super().__init__()
        self.heads = heads
        self.ln1 = torch.nn.LayerNorm(dim)
        self.qkv = fc(dim, 3 * dim, seeds())
        self.proj = fc(dim, dim, seeds())
        self.ln2 = torch.nn.LayerNorm(dim)
        self.fc = fc(dim, 4 * dim, seeds())
        self.fc_proj = fc(4 * dim, dim, seeds())

    def forward(self, x):
        x = x + mha(self.ln1(x), self.qkv, self.proj, self.heads, None, True)
        return x + self.fc_proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh"))


class GPT2(torch.nn.Module):
    def __init__(self, vocab, context, dim, heads, layers, seed):
        super().__init__()
        seeds = Seeds(seed)
        self.tokens = torch.nn.Parameter(uniform_std(seeds(), 0.02, vocab, dim))
        self.positions = torch.nn.Parameter(uniform_std(seeds(), 0.02, context, dim))
        self.blocks = torch.nn.ModuleList([GPT2Block(dim, heads, seeds) for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, idx):
        x = self.tokens[idx] + self.positions[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.norm(x) @ self.tokens.T


def rope(x, base):
    t, d = x.shape[-2], x.shape[-1]
    ang = torch.outer(torch.arange(t, dtype=torch.float64), torch.tensor([base ** (-2.0 * i / d) for i in range(d // 2)], dtype=torch.float64))
    c, s = torch.cos(ang).float().to(x.dtype), torch.sin(ang).float().to(x.dtype)   # tables stored in float32, like jitNN
    a, b = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([a * c - b * s, a * s + b * c], dim=-1)


class LlamaBlock(torch.nn.Module):
    def __init__(self, dim, heads, hidden, base, seeds):
        super().__init__()
        self.heads, self.base = heads, base
        self.attn_norm = torch.nn.RMSNorm(dim, eps=1e-5)
        self.wq = fc(dim, dim, seeds(), False)
        self.wk = fc(dim, dim, seeds(), False)
        self.wv = fc(dim, dim, seeds(), False)
        self.wo = fc(dim, dim, seeds(), False)
        self.ffn_norm = torch.nn.RMSNorm(dim, eps=1e-5)
        self.w1 = fc(dim, hidden, seeds(), False)
        self.w3 = fc(dim, hidden, seeds(), False)
        self.w2 = fc(hidden, dim, seeds(), False)

    def forward(self, x):
        b, t, d = x.shape
        n = self.attn_norm(x)
        heads = lambda y: y.reshape(b, t, self.heads, -1).permute(0, 2, 1, 3)
        q = rope(heads(self.wq(n)), self.base)
        k = rope(heads(self.wk(n)), self.base)
        v = heads(self.wv(n))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(y.permute(0, 2, 1, 3).reshape(b, t, d))
        m = self.ffn_norm(x)
        return x + self.w2(F.silu(self.w1(m)) * self.w3(m))


class Llama(torch.nn.Module):
    def __init__(self, vocab, dim, heads, layers, hidden, base, seed):
        super().__init__()
        seeds = Seeds(seed)
        self.tokens = torch.nn.Parameter(uniform_std(seeds(), 0.02, vocab, dim))
        self.blocks = torch.nn.ModuleList([LlamaBlock(dim, heads, hidden, base, seeds) for _ in range(layers)])
        self.norm = torch.nn.RMSNorm(dim, eps=1e-5)
        self.output = fc(dim, vocab, seeds(), False)

    def forward(self, idx):
        x = self.tokens[idx]
        for b in self.blocks:
            x = b(x)
        return self.output(self.norm(x))


class BertBlock(torch.nn.Module):
    def __init__(self, dim, heads, seeds):
        super().__init__()
        self.heads = heads
        self.qkv = fc(dim, 3 * dim, seeds())
        self.proj = fc(dim, dim, seeds())
        self.ln1 = torch.nn.LayerNorm(dim, eps=1e-12)
        self.fc1 = fc(dim, 4 * dim, seeds())
        self.fc2 = fc(4 * dim, dim, seeds())
        self.ln2 = torch.nn.LayerNorm(dim, eps=1e-12)

    def forward(self, x, mask):
        x = self.ln1(x + mha(x, self.qkv, self.proj, self.heads, mask, False))
        return self.ln2(x + self.fc2(F.gelu(self.fc1(x))))


class Bert(torch.nn.Module):
    def __init__(self, vocab, context, dim, heads, layers, seed):
        super().__init__()
        seeds = Seeds(seed)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(vocab))
        self.tokens = torch.nn.Parameter(uniform_std(seeds(), 0.02, vocab, dim))
        self.positions = torch.nn.Parameter(uniform_std(seeds(), 0.02, context, dim))
        self.emb_norm = torch.nn.LayerNorm(dim, eps=1e-12)
        self.blocks = torch.nn.ModuleList([BertBlock(dim, heads, seeds) for _ in range(layers)])
        self.head_dense = fc(dim, dim, seeds())
        self.head_norm = torch.nn.LayerNorm(dim, eps=1e-12)

    def forward(self, idx, mask, rows):
        x = self.emb_norm(self.tokens[idx] + self.positions[: idx.shape[1]])
        for b in self.blocks:
            x = b(x, mask)
        z = self.head_norm(F.gelu(self.head_dense(x.reshape(-1, x.shape[-1])[rows])))
        return z @ self.tokens.T + self.decoder_bias


def build(kind):
    if kind == "gpt2":
        return GPT2(VOCAB, SEQ, DIM, HEADS, LAYERS, 1)
    if kind == "llama":
        return Llama(VOCAB, DIM, HEADS, LAYERS, LLAMA_HIDDEN, 10000.0, 1)
    return Bert(BERT_VOCAB, SEQ, DIM, HEADS, LAYERS, 1)


# ---- data, laid out exactly as jitnn.data.TextCorpus and TransformerAccelTest

def load(path):
    tokens = np.frombuffer(open(path, "rb").read(), dtype=np.uint8).astype(np.int64)
    return tokens, len(tokens) * 9 // 10


def lm_batch(tokens, split, b):
    starts = [((b * BATCH + r) * SEQ) % (split - SEQ - 1) for r in range(BATCH)]
    x = np.stack([tokens[s:s + SEQ] for s in starts])
    y = np.stack([tokens[s + 1:s + SEQ + 1] for s in starts])
    return torch.from_numpy(x), torch.from_numpy(y)


def mlm_hash(key, r, p):
    return ((((key * BATCH + r) * SEQ + p) * 2654435761) & 0xFFFFFFFF)


def mlm_batch(x, key):
    """Pads row r to length SEQ - (key_r * 7) % 33 and masks PREDICTIONS of its real positions, those
    with the smallest integer hash (a fixed count keeps the shapes, and the compiled step, the same);
    returns inputs, an additive mask [B, 1, 1, T], the flat masked rows and their labels."""
    x = x.numpy().copy()
    mask = np.zeros((BATCH, 1, 1, SEQ), np.float32)
    rows, labels = [], []
    for r in range(BATCH):
        length = SEQ - ((key * BATCH + r) * 7) % 33
        chosen = sorted(sorted(range(length), key=lambda p: (mlm_hash(key, r, p), p))[:PREDICTIONS])
        for p in chosen:
            rows.append(r * SEQ + p)
            labels.append(x[r, p])
            x[r, p] = MASK
        for p in range(length, SEQ):
            x[r, p] = PAD
            mask[r, 0, 0, p] = -1e9
    return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(rows), torch.tensor(labels)


def loss_of(kind, net, tokens, split, b, dtype, dev):
    x, y = lm_batch(tokens, split, b)
    if kind == "bert":
        xi, mask, rows, labels = mlm_batch(x, b)
        return F.cross_entropy(net(xi.to(dev), mask.to(dtype).to(dev), rows.to(dev)), labels.to(dev))
    logits = net(x.to(dev))
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).to(dev))


def eval_loss(kind, net, tokens, split, dev):
    windows = (len(tokens) - split - 1) // SEQ
    total, count = 0.0, 0
    with torch.no_grad():
        for w0 in range(0, windows - BATCH + 1, BATCH):
            starts = [split + (w0 + r) * SEQ for r in range(BATCH)]
            x = torch.from_numpy(np.stack([tokens[s:s + SEQ] for s in starts]))
            y = torch.from_numpy(np.stack([tokens[s + 1:s + SEQ + 1] for s in starts]))
            if kind == "bert":
                xi, mask, rows, labels = mlm_batch(x, 1_000_000 + w0)
                total += F.cross_entropy(net(xi.to(dev), mask.to(dev), rows.to(dev)), labels.to(dev)).item()
            else:
                logits = net(x.to(dev))
                total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1).to(dev)).item()
            count += 1
    return total / count


def schedule(step, total):
    if step < WARMUP:
        return step / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return MIN_RATIO + (1 - MIN_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))


def train(kind, corpus, out, steps):
    torch.set_num_threads(8)
    dev = torch.device(os.environ.get("JITNN_TORCH_DEVICE", "cpu"))
    tokens, split = load(corpus)
    net = build(kind).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: schedule(s, steps))
    losses = []
    net.train()
    for b in range(steps):
        loss = loss_of(kind, net, tokens, split, b, torch.float32, dev)
        opt.zero_grad()
        loss.backward()
        if b == 0:
            np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in net.parameters()]).astype(np.float32).tofile(out + ".grad0")
        torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP)
        opt.step()
        sched.step()
        losses.append(loss.item())
    net.eval()
    ev = eval_loss(kind, net, tokens, split, dev)
    np.array([ev] + losses, dtype=np.float32).tofile(out)
    print(f"{kind}: {steps} steps, loss {losses[0]:.4f} -> {np.mean(losses[-20:]):.4f}, eval loss {ev:.4f}")


def grad64(kind, corpus, prefix):
    dev = torch.device(os.environ.get("JITNN_TORCH_DEVICE", "cpu"))
    tokens, split = load(corpus)
    grads = {}
    for dtype in (torch.float64, torch.float32):
        net = build(kind).to(dtype).to(dev)
        net.train()
        loss_of(kind, net, tokens, split, 0, dtype, dev).backward()
        grads[dtype] = [p.grad.detach().double().cpu().numpy().ravel() for p in net.parameters()]
    g64, g32 = grads[torch.float64], grads[torch.float32]
    err = [np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30) for a, b in zip(g64, g32)]
    np.concatenate(g64).astype(np.float64).tofile(prefix + ".grad0_f64")
    np.array(err, dtype=np.float32).tofile(prefix + ".grad0_f32err")
    print(f"{kind}: {len(err)} parameters, PyTorch FP32 vs FP64 relative error: max {max(err):.2e}, median {np.median(err):.2e}")


if __name__ == "__main__":
    mode, kind = sys.argv[1], sys.argv[2]
    if mode == "train":
        train(kind, sys.argv[3], sys.argv[4], int(sys.argv[5]))
    else:
        grad64(kind, sys.argv[3], sys.argv[4])
