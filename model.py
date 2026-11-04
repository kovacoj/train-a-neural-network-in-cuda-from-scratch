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

# Step 2 - device_upload
#include <cuda_runtime.h>

float* device_upload(const float* host, int n) {
    // TODO: cudaMalloc n floats, cudaMemcpy host -> device, return the device pointer
    float* dev = nullptr; cudaMalloc(&dev, sizeof(float) * n);
    cudaMemcpy(dev, host, sizeof(float) * n, cudaMemcpyHostToDevice);

    return dev;
}

int* device_upload(const int* host, int n) {
    // TODO: same for ints
    int* dev = nullptr; cudaMalloc(&dev, sizeof(int) * n);
    cudaMemcpy(dev, host, sizeof(int) * n, cudaMemcpyHostToDevice);

    return dev;
}

void device_download(float* host, const float* dev, int n) {
    // TODO: cudaMemcpy device -> host
    cudaMemcpy(host, dev, sizeof(float) * n, cudaMemcpyDeviceToHost);
}

# Step 3 - make_spiral
#include <cmath>

float lcg_uniform(unsigned* state) {
    // TODO: *state = *state * 1664525u + 1013904223u; return (*state >> 8) / 16777216.0f
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) / 16777216.0f;
}

float lcg_normal(unsigned* state) {
    // TODO: Box-Muller from two lcg_uniform draws (u1 first, then u2)
    float u1 = lcg_uniform(state), u2 = lcg_uniform(state);
    return sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(6.2831853f * u2);
}

void make_spiral(float* X, int* y, int points_per_class, int classes, float noise, unsigned seed) {
    // TODO: one generator seeded with `seed`; for each class c, for each point i:
    // r = i / (points_per_class - 1), t = 4c + 4r + noise * normal, X = (r sin t, r cos t), y = c
    float r, t; int k;
    unsigned state = seed;
    for (int c=0; c < classes; c++) {
        for (int i=0; i < points_per_class; i++) {
            r = (float)i / (float)(points_per_class - 1);
            t = 4.0f * c + 4.0f * r + noise * lcg_normal(&state);
            k = c * points_per_class + i;
            X[2*k] = r * sinf(t); X[2*k + 1] = r * cosf(t);
            y[k] = c;
        }
    }
}

# Step 4 - matmul_kernel
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: col from x, row from y, guard, accumulate over k, store C[row * N + col]
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k=0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: 16 x 16 blocks, grid covering N columns (x) and M rows (y)
    dim3 block(16, 16);
    dim3 threads((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<threads, block>>>(A, B, C, M, N, K);
}

# Step 5 - add_bias_kernel
#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    // TODO: one thread per element; element i gets b[i % cols]
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows * cols) {
        Z[i] += b[i % cols];
    }
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    // TODO: 256 threads per block over rows * cols elements
    int threads = 256;
    int blocks = (rows * cols + threads - 1) / 256;

    add_bias_kernel<<<blocks, threads>>>(Z, b, rows, cols);
}

# Step 6 - relu_kernel
#include <cuda_runtime.h>

__global__ void relu_kernel(float* Z, int n) {
    // TODO: Z[i] = max(Z[i], 0) for i < n
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        Z[i] = Z[i] > 0 ? Z[i] : 0.0f;
    }
}

void launch_relu(float* Z, int n) {
    // TODO: 256 threads per block
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_kernel<<<blocks, threads>>>(Z, n);
}

# Step 7 - softmax_rows_kernel
#include <cuda_runtime.h>

__global__ void softmax_rows_kernel(const float* logits, float* probs, int rows, int cols) {
    // TODO: one thread per row: max, sum of exp(z - max), then normalise
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows) {
    r *= cols;
    float max = logits[r];
        for (int i=1; i < cols; i++) {
            max = max > logits[r + i] ? max : logits[r + i];
        }

        float sum = 0.0f;
        for (int i=0; i < cols; i++) {
            sum += expf(logits[r + i] - max);
        }

        for (int i=0; i < cols; i++) {
            probs[r + i] = expf(logits[r + i] - max) / sum;
        }
    }
}

void launch_softmax_rows(const float* logits, float* probs, int rows, int cols) {
    // TODO: 128 threads per block, one thread per row
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;

    softmax_rows_kernel<<<blocks, threads>>>(logits, probs, rows, cols);
}

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

