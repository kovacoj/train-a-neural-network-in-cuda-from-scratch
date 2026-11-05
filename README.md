# Two-Layer MLP in CUDA from Scratch

A small neural network trained entirely with hand-written CUDA kernels. The
program learns a nonlinear, three-class spiral dataset using a `2 -> 64 -> 3`
multilayer perceptron, full-batch gradient descent, and no machine-learning or
linear-algebra libraries.

The implementation includes the forward pass, cross-entropy loss,
backpropagation, SGD updates, evaluation, an ASCII decision-boundary plot, and
a comparison with a serial CPU reference.

## Recorded result

The following result was recorded on an NVIDIA Tesla T4. It has not yet been
reproduced in CI or on another GPU.

| Configuration | Value |
|---|---:|
| GPU | Tesla T4, 40 SMs |
| Network | `2 -> 64 -> 3` |
| Training set | 600 points |
| Held-out set | 300 points |
| Optimizer | Full-batch SGD |
| Learning rate | 1.0 |
| Epochs | 2,000 |
| Training seed | 1 |
| Held-out seed | 2 |
| Parameter seed | 7 |

| Metric | Training | Held-out |
|---|---:|---:|
| Accuracy | 99.7% | 99.3% |
| Reported loss | 0.0331 | 0.0379 |

### Decision regions

The regions below are predictions from the trained CUDA model. `A`, `B`, and
`C` mark every fourth training point; `.`, `-`, and a space represent the
three predicted classes.

```text
  |                          --------------------------------------|
  |                          --------------------------------------|
  |                     C    --------------------------------------|
  |                C        ---------------------------------------|
  |              C         ----------------------------------------|
  |               C     -----------B-BB------B---B-----------------|
  |          CC       ---------B-B---B----B-----B----B-------------|
  |                  ---------BB---------B-------B-B-B-------------|
  |                  ------B-BB------.........---------------------|
  |         CC      ------BBB-----...............-----------BB-----|
  |          C      ------BBB----....AAAA..A........----------B----|
  |        C        --------BBB---AAAA...AAAAAAA.....------BB------|
  |          C CC    --------BBBBBBCCCC  ....A.AA......----B---B---|
  |              C     ---------     CCC  ......AA......-----------|
  |            C C                  CC     ............------------|
  |                  C C      C    CC      ...A.AA.....------------|
  |.............  C   C  C  C CCC        .....A..A.....------------|
  |...............        C          ........A............---------|
  |..........A.A........  ................A..A...............------|
  |.................AA............A...A.A.A.....................---|
  |.................AA....A.A..AA......A...........................|
  |................................................................|
  |................................................................|
  |................................................................|
```

## Build and run

### Requirements

- A CUDA-capable NVIDIA GPU
- An NVIDIA driver compatible with the installed CUDA toolkit
- CUDA toolkit with `nvcc`
- A C++17-compatible host compiler

Build the canonical source file directly:

```bash
nvcc -std=c++17 -O2 model.cu -o cuda-mlp
./cuda-mlp
```

The exported `model.py` and `scaffold.py` files are retained as Deep-ML
artifacts. They contain CUDA/C++ source and are not executable Python. The
standalone program is `model.cu`.

## Model

For a batch `X` with labels `y`, the forward pass is:

```text
H = ReLU(X W1 + b1)
P = softmax(H W2 + b2)
L = mean(-log(P[row, y[row]]))
```

Backpropagation computes:

```text
dZ2 = (P - one_hot(y)) / batch
dW2 = H^T dZ2
db2 = column_sum(dZ2)
dH  = (dZ2 W2^T) * (H > 0)
dW1 = X^T dH
db1 = column_sum(dH)
```

All arrays are row-major device buffers:

| Buffer | Shape |
|---|---:|
| `X` | `[batch, 2]` |
| `W1` | `[2, 64]` |
| `b1` | `[64]` |
| `H` | `[batch, 64]` |
| `W2` | `[64, 3]` |
| `b2` | `[3]` |
| `P` | `[batch, 3]` |

Each operation is implemented as a CUDA kernel: matrix multiplication, bias
broadcasting, ReLU, stable row-wise softmax, cross-entropy reduction,
backpropagation, column sums, parameter updates, and accuracy.

## Performance

Recorded timings from the Tesla T4 run:

| Points | GPU ms/step | Serial CPU ms/step | Reported speedup |
|---:|---:|---:|---:|
| 600 | 0.122 | 3.855 | 31.7x |
| 12,000 | 1.129 | 76.554 | 67.8x |
| 120,000 | 15.818 | 779.431 | 49.3x |

These numbers compare the CUDA implementation with a hand-written,
single-threaded CPU reference, not optimized BLAS. They are useful for showing
how the same direct loops map to the GPU, but they are not a controlled
CPU-versus-GPU hardware comparison:

- The CPU reference allocates temporary vectors during each step.
- The CPU reference does not calculate the scalar loss.
- The GPU path retrieves the scalar loss and synchronizes with the host.
- Compiler version, CUDA version, CPU model, and compiler flags were not
  recorded for this run.

The GPU timing uses CUDA events after one warm-up step. Repetition counts are
200, 50, and 20 for batches of 600, 12,000, and 120,000 points respectively.
CPU repetition counts are 20, 3, and 1.

## Limitations

The kernels favor clarity over performance. Matrix multiplications are
untiled, operations are not fused, and all training is full-batch FP32 SGD.
The weight and bias gradients assign one thread to each output element, so at
large batch sizes only 3 to 192 threads perform each reduction while most of
the GPU is idle. Parallel reductions, tiled matrix multiplication, and kernel
fusion are the most direct optimization opportunities.

The repository does not currently include standalone numerical correctness
tests or automated GPU CI. Most CUDA runtime calls and kernel launches also
need consistent error checking before treating this as production-quality
code.

## Example run

<details>
<summary>Complete recorded Tesla T4 output</summary>

```text
GPU: Tesla T4, 40 SMs

Before training: train accuracy 0.268 (chance is 0.333)

Training 2 -> 64 -> 3, lr 1.0, 2000 full-batch epochs:
epoch     1 loss 1.1876
epoch   250 loss 0.1178
epoch   500 loss 0.0751
epoch   750 loss 0.0584
epoch  1000 loss 0.0492
epoch  1250 loss 0.0432
epoch  1500 loss 0.0389
epoch  1750 loss 0.0357
epoch  2000 loss 0.0331

Final: train loss 0.0331, train accuracy 0.997 | held-out loss 0.0379, held-out accuracy 0.993

Decision regions ('.' = class 0, '-' = class 1, ' ' = class 2; every 4th training point as A/B/C):
  |                          --------------------------------------|
  |                          --------------------------------------|
  |                     C    --------------------------------------|
  |                C        ---------------------------------------|
  |              C         ----------------------------------------|
  |               C     -----------B-BB------B---B-----------------|
  |          CC       ---------B-B---B----B-----B----B-------------|
  |                  ---------BB---------B-------B-B-B-------------|
  |                  ------B-BB------.........---------------------|
  |         CC      ------BBB-----...............-----------BB-----|
  |          C      ------BBB----....AAAA..A........----------B----|
  |        C        --------BBB---AAAA...AAAAAAA.....------BB------|
  |          C CC    --------BBBBBBCCCC  ....A.AA......----B---B---|
  |              C     ---------     CCC  ......AA......-----------|
  |            C C                  CC     ............------------|
  |                  C C      C    CC      ...A.AA.....------------|
  |.............  C   C  C  C CCC        .....A..A.....------------|
  |...............        C          ........A............---------|
  |..........A.A........  ................A..A...............------|
  |.................AA............A...A.A.A.....................---|
  |.................AA....A.A..AA......A...........................|
  |................................................................|
  |................................................................|
  |................................................................|

One training step, GPU vs CPU (2 -> 64 -> 3, full batch):
       600 points   GPU    0.122 ms/step   CPU     3.855 ms/step   GPU is   31.7x faster
     12000 points   GPU    1.129 ms/step   CPU    76.554 ms/step   GPU is   67.8x faster
    120000 points   GPU   15.818 ms/step   CPU   779.431 ms/step   GPU is   49.3x faster

The CPU column is one core running the same loops. At large batches most of the GPU's time goes to the four
kernels that sum over the batch (dW1, db1, dW2, db2): one thread per output means only 3 to 192 threads, each
walking every row while the rest of the GPU idles. Splitting those sums across many blocks (a reduction like
your cross-entropy kernel), fusing kernels and tiling the matrix products are what CUDA Intermediate covers.
```

</details>

## Attribution

Originally completed as the Deep-ML **Train a Neural Network in CUDA from
Scratch** project. This repository preserves the from-scratch implementation
while making the generated CUDA program available as a standalone project.
