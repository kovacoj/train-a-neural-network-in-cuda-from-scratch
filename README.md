# Train a Neural Network in CUDA from Scratch

Train a real neural network on the GPU using nothing but the kernels you met in CUDA Basics. Start with the plumbing every CUDA program needs: an error check, device buffers, and a deterministic three-class spiral dataset that no straight line can separate. Then write the forward pass as small kernels (a naive matrix multiply, a bias broadcast, ReLU, a numerically stable row softmax, and a cross-entropy loss built on a shared-memory reduction), derive the backward pass and write it as kernels too (the softmax gradient p - y, the two transposed matrix products of the chain rule, column sums for the biases, the ReLU mask), check those gradients against a CPU reference and finite differences, and wire everything into a gradient-descent training loop. The finished program trains a two-layer MLP to about 99% accuracy on held-out points, draws its decision regions in the terminal, and times a training step on the GPU against the CPU. Every kernel is one thread per output: no tiling, no libraries, so you can see exactly where the time goes and what CUDA Intermediate will speed up next.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** report_cuda_error
- [x] **2.** device_upload
- [x] **3.** make_spiral
- [x] **4.** matmul_kernel
- [x] **5.** add_bias_kernel
- [x] **6.** relu_kernel
- [x] **7.** softmax_rows_kernel
- [x] **8.** cross_entropy_kernel
- [x] **9.** init_mlp
- [x] **10.** alloc_workspace
- [x] **11.** forward_mlp
- [x] **12.** softmax_ce_grad_kernel
- [x] **13.** matmul_tn_kernel
- [x] **14.** matmul_nt_kernel
- [x] **15.** column_sum_kernel
- [x] **16.** relu_backward_kernel
- [x] **17.** backward_mlp
- [x] **18.** sgd_update_kernel
- [x] **19.** train_step
- [x] **20.** train_mlp
- [x] **21.** accuracy_kernel
- [x] **22.** time_train_step_ms

## Run
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
       600 points   GPU    0.104 ms/step   CPU     3.846 ms/step   GPU is   36.8x faster
     12000 points   GPU    1.233 ms/step   CPU    75.758 ms/step   GPU is   61.5x faster
    120000 points   GPU   16.124 ms/step   CPU   744.368 ms/step   GPU is   46.2x faster

The CPU column is one core running the same loops. At large batches most of the GPU's time goes to the four
kernels that sum over the batch (dW1, db1, dW2, db2): one thread per output means only 3 to 192 threads, each
walking every row while the rest of the GPU idles. Splitting those sums across many blocks (a reduction like
your cross-entropy kernel), fusing kernels and tiling the matrix products are what CUDA Intermediate covers.

---

Built on Deep-ML.
