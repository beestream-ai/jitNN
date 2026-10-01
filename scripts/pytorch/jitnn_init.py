"""jitNN's counter-based initialisation in numpy, so PyTorch can start from bit-identical weights.

Philox4x32-10 with jitNN's layout: element i takes word (i & 3) of the block with counter
(i >> 2, stream, 0, 0) and key (seed & 0xffffffff, seed >> 32).
"""
import numpy as np

M0, M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
W0, W1 = np.uint32(0x9E3779B9), np.uint32(0xBB67AE85)
MASK32 = np.uint64(0xFFFFFFFF)


def philox(c0, c1, c2, c3, k0, k1, rounds=10):
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint32) for c in (c0, c1, c2, c3))
    k0, k1 = np.uint32(k0), np.uint32(k1)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            p0 = M0 * c0.astype(np.uint64)
            p1 = M1 * c2.astype(np.uint64)
            hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & MASK32).astype(np.uint32)
            hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & MASK32).astype(np.uint32)
            c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
            k0 = np.uint32(k0 + W0)
            k1 = np.uint32(k1 + W1)
    return c0, c1, c2, c3


def uniform(seed, lo, hi, n):
    seed &= (1 << 64) - 1
    i = np.arange(n, dtype=np.uint32)
    words = philox(i >> np.uint32(2), np.zeros(n, np.uint32), 0, 0, seed & 0xFFFFFFFF, seed >> 32)
    w = np.choose((i & np.uint32(3)).astype(np.int64), words)
    unit = ((w >> np.uint32(8)).astype(np.float32) * np.float32(5.9604645e-8))
    return np.float32(lo) + (np.float32(hi) - np.float32(lo)) * unit


def linear(in_features, out_features, seed):
    """Weight [out, in] and bias [out] exactly as jitnn.nn.Linear(in, out, seed) creates them."""
    bound = np.float32(1.0 / np.sqrt(in_features))
    w = uniform(seed, -bound, bound, out_features * in_features).reshape(out_features, in_features)
    b = uniform(seed + 0x9E3779B97F4A7C15, -bound, bound, out_features)
    return w, b


if __name__ == "__main__":
    assert [hex(int(x)) for x in philox(0, 0, 0, 0, 0, 0)] == ["0x6627e8d5", "0xe169c58d", "0xbc57ac4c", "0x9b00dbd8"]
    assert [hex(int(x)) for x in philox(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0)] == \
        ["0xd16cfe09", "0x94fdcceb", "0x5001e420", "0x24126ea1"]
    print("philox KATs ok; uniform(42,-1,3)[:4] =", uniform(42, -1, 3, 4))
