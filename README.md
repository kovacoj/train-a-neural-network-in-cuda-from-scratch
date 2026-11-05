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

---

Built on Deep-ML.
