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

# Step 8 - cross_entropy_kernel
#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* probs, const int* labels, int rows, int cols, float* total) {
    // TODO: per-thread -log(max(p[label], 1e-12)), shared-memory tree sum of 256 values,
    // thread 0 atomicAdds the block sum into *total
    __shared__ float sums[256];

    int tid = threadIdx.x;
    int r = blockIdx.x * blockDim.x + tid;

    sums[tid] = r >= rows ? 0.0f : -logf(fmaxf(probs[r * cols + labels[r]], 1e-12f));

    __syncthreads();

    for (int diff=128; diff > 0; diff >>= 1) {
        if (tid < diff) {
            sums[tid] += sums[tid + diff];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total, sums[0]);
    }
}

float cross_entropy_loss(const float* probs, const int* labels, int rows, int cols) {
    // TODO: zeroed device float, launch 256-thread blocks, copy back, free, return total / rows
    float* dtotal = nullptr; float total;
    cudaMalloc(&dtotal, sizeof(float));
    cudaMemset(dtotal, 0, sizeof(float));
    
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    cross_entropy_kernel<<<blocks, threads>>>(probs, labels, rows, cols, dtotal);
    cudaMemcpy(&total, dtotal, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dtotal);

    return total / rows;
}

# Step 9 - init_mlp
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

struct MLP {
    int in_dim, hidden, out_dim;
    float *W1, *b1, *W2, *b2;   // device pointers
};

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed) {
    MLP m = {in_dim, hidden, out_dim, nullptr, nullptr, nullptr, nullptr};
    // TODO: one generator seeded with `seed`; W1 then W2, (2u - 1) * sqrt(6 / fan_in); zero biases;
    // upload all four with device_upload
    unsigned state = seed;
    float* W1 = (float*)malloc(in_dim * hidden * sizeof(float));
    float* W2 = (float*)malloc(hidden * out_dim * sizeof(float));
    float* b1 = (float*)malloc(hidden * sizeof(float));
    float* b2 = (float*)malloc(out_dim * sizeof(float));

    for (int col=0; col < hidden; col++) b1[col] = 0.0f;
    for (int col=0; col < out_dim; col++) b2[col] = 0.0f;

    for (int row=0; row < in_dim; row++) {
        for (int col=0; col < hidden; col++) {
            W1[row * hidden + col] = (2 * lcg_uniform(&state) - 1) * sqrtf(6.0f / in_dim);
        }
    }

    for (int row=0; row < hidden; row++) {
        for (int col=0; col < out_dim; col++) {
            W2[row * out_dim + col] = (2 * lcg_uniform(&state) - 1) * sqrtf(6.0f / hidden);
        }
    }

    m.W1 = device_upload(W1, in_dim * hidden); free(W1);
    m.b1 = device_upload(b1, hidden); free(b1);
    m.W2 = device_upload(W2, hidden * out_dim); free(W2);
    m.b2 = device_upload(b2, out_dim); free(b2);

    return m;
}

void free_mlp(MLP& m) {
    // TODO: cudaFree the four arrays, set them to nullptr
    cudaFree(m.W1); cudaFree(m.W2); cudaFree(m.b1); cudaFree(m.b2);
    m.W1 = m.W2 = m.b1 = m.b2 = nullptr;
}

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

# Step 15 - column_sum_kernel
#include <cuda_runtime.h>

__global__ void column_sum_kernel(const float* D, float* out, int rows, int cols) {
    // TODO: thread c sums column c over all rows
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < cols) {
        float sum = 0.0f;

        for (int r=0; r < rows; r++) {
            sum += D[r * cols + c];
        }

        out[c] = sum;
    }
}

void launch_column_sum(const float* D, float* out, int rows, int cols) {
    // TODO: 128 threads per block, one per column
    int threads = 128;
    int blocks = (cols + threads - 1) / threads;

    column_sum_kernel<<<blocks, threads>>>(D, out, rows, cols);
}

# Step 16 - relu_backward_kernel
#include <cuda_runtime.h>

__global__ void relu_backward_kernel(float* dH, const float* H, int n) {
    // TODO: dH[i] stays where H[i] > 0, else 0
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        dH[i] = H[i] > 0 ? dH[i] : 0.0f;
    }
}

void launch_relu_backward(float* dH, const float* H, int n) {
    // TODO: 256 threads per block
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(dH, H, n);
}

# Step 17 - backward_mlp (not yet solved)
# TODO: implement

# Step 18 - sgd_update_kernel
#include <cuda_runtime.h>

__global__ void sgd_update_kernel(float* param, const float* grad, float lr, int n) {
    // TODO: param[i] -= lr * grad[i]
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if  (i < n) {
        param[i] -= lr * grad[i];
    }
}

void launch_sgd_update(float* param, const float* grad, float lr, int n) {
    // TODO: 256 threads per block
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    sgd_update_kernel<<<blocks, threads>>>(param, grad, lr, n);
}

# Step 19 - train_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - accuracy_kernel (not yet solved)
# TODO: implement

# Step 22 - time_train_step_ms (not yet solved)
# TODO: implement

