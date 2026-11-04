"""
Train a Neural Network in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - report_cuda_error
#include <cstdio>
#include <cuda_runtime.h>

int report_cuda_error(cudaError_t err, const char* what) {
    // TODO: return 0 on cudaSuccess; otherwise print
    // "CUDA error in <what>: <cudaGetErrorString(err)>" to stderr and return 1
    if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in %s: %s\n",
            what, cudaGetErrorString(err));
    }
    return err == cudaSuccess ? 0 : 1;
}

# Step 2 - device_upload (not yet solved)
# TODO: implement

# Step 3 - make_spiral (not yet solved)
# TODO: implement

# Step 4 - matmul_kernel (not yet solved)
# TODO: implement

# Step 5 - add_bias_kernel (not yet solved)
# TODO: implement

# Step 6 - relu_kernel (not yet solved)
# TODO: implement

# Step 7 - softmax_rows_kernel (not yet solved)
# TODO: implement

# Step 8 - cross_entropy_kernel (not yet solved)
# TODO: implement

# Step 9 - init_mlp (not yet solved)
# TODO: implement

# Step 10 - alloc_workspace (not yet solved)
# TODO: implement

# Step 11 - forward_mlp (not yet solved)
# TODO: implement

# Step 12 - softmax_ce_grad_kernel (not yet solved)
# TODO: implement

# Step 13 - matmul_tn_kernel (not yet solved)
# TODO: implement

# Step 14 - matmul_nt_kernel (not yet solved)
# TODO: implement

# Step 15 - column_sum_kernel (not yet solved)
# TODO: implement

# Step 16 - relu_backward_kernel (not yet solved)
# TODO: implement

# Step 17 - backward_mlp (not yet solved)
# TODO: implement

# Step 18 - sgd_update_kernel (not yet solved)
# TODO: implement

# Step 19 - train_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - accuracy_kernel (not yet solved)
# TODO: implement

# Step 22 - time_train_step_ms (not yet solved)
# TODO: implement

