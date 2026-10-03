# Validation

Four layers of checks run on the host and on every device.

1. **Host reference.** Every operator has a double-precision Java implementation, written
   independently of its kernel. `Engine.Backend.REFERENCE` runs it, and it is the oracle the device
   tests compare against.
2. **Finite differences.** `GradientTest` compares every gradient with central differences:
   broadcasting binaries, every unary, reductions, all four GEMM transposes, permute, reshape,
   softmax, log-softmax, cross-entropy, fused linear and an MLP chain. This currently runs in FP32
   with a loose tolerance, until jitNN has an FP64 type.
3. **PyTorch goldens.** `scripts/pytorch/goldens.py` computes 104 operator cases (wave 1 and the
   convolution, BatchNorm, pooling and ReLU6 cases of wave 2), in FP64 (inputs,
   output and every input gradient, on odd, size-1 and broadcast shapes), plus 5 optimiser cases
   (three `torch.optim` steps each), and Philox values reproduced in numpy. `GoldenTest` checks the
   reference against them; `GoldenAccelTest` and `OptimGoldenAccelTest` check every device.
4. **Model parity.** Models start from bit-identical initial weights (`scripts/pytorch/jitnn_init.py`
   reimplements jitNN's Philox initialisation; `models.py` builds ResNet-18 and MobileNetV2 with the
   same seeds) and train on the same batches as PyTorch.
   - **MNIST gates** (MLP, 2 epochs; LeNet-5, 1 epoch): losses of steps 0-9 within 1e-4, steps
     0-49 within 1%, the mean of the last 100 within 5%, accuracy within 0.3 points.
   - **CNN gates** (ResNet-18 and MobileNetV2: 200 steps on MNIST, and one epoch of 781 steps on
     CIFAR-10 with evaluation on all 10,000 test images): step-by-step loss curves of deep
     BatchNorm networks drift apart within a few Adam steps from FP32 rounding alone, so the gate
     compares every parameter's step-0 gradient with PyTorch in FP64 (`cnn_grad64.py`). A parameter
     passes within three times PyTorch's own FP32 error on it, relative to its gradient or, for
     gradients that nearly cancel to zero, against the model's median absolute FP32 error. Then
     the step-0 loss (1e-5), the mean loss of the last 50 steps (10%) and the evaluation accuracy
     (3 points).

## Results

| Check | Reference | OpenCL (M4 Pro) | Metal (M4 Pro) | CUDA (RTX 5070 Ti) | CUDA (A10) |
|---|---|---|---|---|---|
| Operator goldens (104) | pass | pass | pass | pass | pass |
| Optimiser goldens (5) | pass | pass | pass | pass | pass |
| Philox against numpy | bit-exact | bit-exact | bit-exact | bit-exact | bit-exact |
| MLP gate: loss vs reference at every step, 1e-3 | | pass | pass | pass | pass |
| MNIST: worst relative loss gap, steps 0-49 | | 5.0e-7 | 7.0e-7 | 6.0e-7 | 5.0e-7 |
| MNIST MLP: test accuracy (PyTorch: 96.24%) | | 96.15% | 96.14% | 96.25% | 96.25% |
| LeNet-5: test accuracy (PyTorch: 96.87%) | | 96.93% | 97.02% | 97.00% | 97.00% |
| ResNet-18: worst step-0 gradient, fraction of its bound | | 0.38 | 0.33 | 0.85 | 0.85 |
| ResNet-18: accuracy after 200 steps (PyTorch: 93.2%) | | 93.0% | 92.7% | 93.9% | 93.9% |
| MobileNetV2: worst step-0 gradient, fraction of its bound | | 0.64 | 0.60 | | 0.69 |
| MobileNetV2: accuracy after 200 steps (PyTorch: 92.3%) | | 94.0% | 91.5% | | 94.9% |
| CIFAR-10 ResNet-18: worst step-0 gradient, fraction of its bound | | 0.23 | 0.26 | | 0.14 |
| CIFAR-10 ResNet-18: mean loss of the last 50 steps (PyTorch: 0.946) | | 0.951 | 0.922 | | 0.945 |
| CIFAR-10 ResNet-18: test accuracy after one epoch (PyTorch: 66.0%) | | 67.5% | 66.5% | | 63.9% |
| CIFAR-10 MobileNetV2: worst step-0 gradient, fraction of its bound | | 0.19 | 0.22 | | 0.24 |
| CIFAR-10 MobileNetV2: mean loss of the last 50 steps (PyTorch: 1.118) | | 1.120 | 1.144 | | 1.137 |
| CIFAR-10 MobileNetV2: test accuracy after one epoch (PyTorch: 59.2%) | | 57.8% | 60.9% | | 58.3% |

The RTX 5070 Ti column predates the CIFAR-10 gates; the A10 column runs every gate.

Tolerances: element-wise ops 2e-6 relative; reductions, GEMM, softmax and losses 1e-4; gradients
ten times those.
