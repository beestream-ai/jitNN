# Validation

Four layers of checks run on the host and on every device.

1. **Host reference.** Every operator has a double-precision Java implementation, written
   independently of its kernel. `Engine.Backend.REFERENCE` runs it, and it is the oracle the device
   tests compare against.
2. **Finite differences.** `GradientTest` compares every gradient with central differences:
   broadcasting binaries, every unary, reductions, all four GEMM transposes, permute, reshape,
   softmax, log-softmax, cross-entropy, fused linear and an MLP chain. This currently runs in FP32
   with a loose tolerance, until jitNN has an FP64 type.
3. **PyTorch goldens.** `scripts/pytorch/goldens.py` computes 87 operator cases in FP64 (inputs,
   output and every input gradient, on odd, size-1 and broadcast shapes), plus 5 optimiser cases
   (three `torch.optim` steps each), and Philox values reproduced in numpy. `GoldenTest` checks the
   reference against them; `GoldenAccelTest` and `OptimGoldenAccelTest` check every device.
4. **Model parity.** The MNIST gate trains a 784-256-128-10 MLP for 2 epochs and runs PyTorch from
   bit-identical initial weights (`scripts/pytorch/jitnn_init.py` reimplements jitNN's Philox
   initialisation) on the same batches.

## Results

| Check | Reference | OpenCL (M4 Pro) | Metal (M4 Pro) | CUDA (RTX 5070 Ti) |
|---|---|---|---|---|
| Operator goldens (87) | pass | pass | pass | pass |
| Optimiser goldens (5) | pass | pass | pass | pass |
| Philox against numpy | bit-exact | bit-exact | bit-exact | bit-exact |
| MLP gate: loss vs reference at every step, 1e-3 | | pass | pass | pass |
| MNIST: worst relative loss gap, steps 0-49 | | 5.0e-7 | 7.0e-7 | 6.0e-7 |
| MNIST: test accuracy (PyTorch: 96.24%) | | 96.15% | 96.14% | 96.14% |

Tolerances: element-wise ops 2e-6 relative; reductions, GEMM, softmax and losses 1e-4; gradients
ten times those.
