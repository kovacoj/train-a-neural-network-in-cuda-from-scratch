# Two-Layer MLP in CUDA from Scratch

A small neural network trained entirely with hand-written CUDA kernels. The
program learns a nonlinear, three-class spiral dataset using a `2 -> 64 -> 3`
multilayer perceptron, full-batch gradient descent, and no machine-learning or
linear-algebra libraries.

The implementation includes the forward pass, cross-entropy loss,
backpropagation, SGD updates, evaluation, an ASCII decision-boundary plot, and
a comparison with a serial CPU reference.

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
