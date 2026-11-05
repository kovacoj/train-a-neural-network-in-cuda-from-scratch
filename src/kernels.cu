#include "kernels.h"
#include <cmath>
#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace {
  __global__ void matmul_kernel(const float* a, const float* b, float* output, int rows,
                                int cols, int inner) {
      const int col = blockIdx.x * blockDim.x + threadIdx.x;
      const int row = blockIdx.y * blockDim.y + threadIdx.y;
      if (row >= rows || col >= cols) {
          return;
      }
  
      float sum = 0.0f;
      for (int index = 0; index < inner; ++index) {
          sum += a[row * inner + index] * b[index * cols + col];
      }
      output[row * cols + col] = sum;
}

__global__ void add_bias_kernel(float* values, const float* bias, int count, int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] += bias[index % cols];
    }
}

__global__ void relu_kernel(float* values, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = values[index] > 0.0f ? values[index] : 0.0f;
    }
}

__global__ void softmax_rows_kernel(const float* logits, float* probabilities, int rows,
                                    int cols) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int offset = row * cols;
    float maximum = logits[offset];
    for (int col = 1; col < cols; ++col) {
        maximum = fmaxf(maximum, logits[offset + col]);
    }

    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
        sum += expf(logits[offset + col] - maximum);
    }
    for (int col = 0; col < cols; ++col) {
        probabilities[offset + col] = expf(logits[offset + col] - maximum) / sum;
    }
}

__global__ void cross_entropy_kernel(const float* probabilities, const int* labels, int rows,
                                     int cols, float* total) {
    __shared__ float sums[256];

    const int thread = threadIdx.x;
    const int row = blockIdx.x * blockDim.x + thread;
    sums[thread] = row < rows
                       ? -logf(fmaxf(probabilities[row * cols + labels[row]], 1e-12f))
                       : 0.0f;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (thread < stride) {
            sums[thread] += sums[thread + stride];
        }
        __syncthreads();
    }

    if (thread == 0) {
        atomicAdd(total, sums[0]);
    }
}

__global__ void softmax_ce_grad_kernel(const float* probabilities, const int* labels,
                                       float* gradient, int rows, int cols) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * cols) {
        return;
    }

    const int row = index / cols;
    const int col = index % cols;
    gradient[index] =
        (probabilities[index] - (col == labels[row] ? 1.0f : 0.0f)) / rows;
}

__global__ void matmul_tn_kernel(const float* a, const float* d, float* output,
                                 int reduction_rows, int a_cols, int d_cols) {
    const int d_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int a_col = blockIdx.y * blockDim.y + threadIdx.y;
    if (a_col >= a_cols || d_col >= d_cols) {
        return;
    }

    float sum = 0.0f;
    for (int row = 0; row < reduction_rows; ++row) {
        sum += a[row * a_cols + a_col] * d[row * d_cols + d_col];
    }
    output[a_col * d_cols + d_col] = sum;
}

__global__ void matmul_nt_kernel(const float* d, const float* weights, float* output,
                                 int rows, int d_cols, int output_cols) {
    const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || output_col >= output_cols) {
        return;
    }

    float sum = 0.0f;
    for (int col = 0; col < d_cols; ++col) {
        sum += d[row * d_cols + col] * weights[output_col * d_cols + col];
    }
    output[row * output_cols + output_col] = sum;
}

__global__ void column_sum_kernel(const float* values, float* output, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }

    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
        sum += values[row * cols + col];
    }
    output[col] = sum;
}

__global__ void relu_backward_kernel(float* gradient, const float* activations, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && activations[index] <= 0.0f) {
        gradient[index] = 0.0f;
    }
}

__global__ void sgd_update_kernel(float* parameter, const float* gradient, float learning_rate,
                                  int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        parameter[index] -= learning_rate * gradient[index];
    }
}

__global__ void accuracy_kernel(const float* probabilities, const int* labels, int rows,
                                int cols, int* correct) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int predicted = 0;
    float maximum = probabilities[row * cols];
    for (int col = 1; col < cols; ++col) {
        if (probabilities[row * cols + col] > maximum) {
            maximum = probabilities[row * cols + col];
            predicted = col;
        }
    }
    if (predicted == labels[row]) {
        atomicAdd(correct, 1);
    }
}
}  // namespace

void launch_matmul(const float* a, const float* b, float* output, int rows, int cols,
                   int inner) {
    const dim3 threads(16, 16);
    const dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    matmul_kernel<<<blocks, threads>>>(a, b, output, rows, cols, inner);
    CUDA_CHECK(cudaGetLastError());
}

void launch_add_bias(float* values, const float* bias, int rows, int cols) {
    constexpr int threads = 256;
    const int count = rows * cols;
    add_bias_kernel<<<(count + threads - 1) / threads, threads>>>(values, bias, count, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu(float* values, int count) {
    constexpr int threads = 256;
    relu_kernel<<<(count + threads - 1) / threads, threads>>>(values, count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_rows(const float* logits, float* probabilities, int rows, int cols) {
    constexpr int threads = 128;
    softmax_rows_kernel<<<(rows + threads - 1) / threads, threads>>>(logits, probabilities, rows,
                                                                    cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cross_entropy(const float* probabilities, const int* labels, int rows, int cols,
                          float* total) {
    constexpr int threads = 256;
    cross_entropy_kernel<<<(rows + threads - 1) / threads, threads>>>(probabilities, labels,
                                                                      rows, cols, total);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax_ce_grad(const float* probabilities, const int* labels, float* gradient,
                            int rows, int cols) {
    constexpr int threads = 256;
    const int count = rows * cols;
    softmax_ce_grad_kernel<<<(count + threads - 1) / threads, threads>>>(
        probabilities, labels, gradient, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_tn(const float* a, const float* d, float* output, int reduction_rows,
                      int a_cols, int d_cols) {
    const dim3 threads(16, 16);
    const dim3 blocks((d_cols + 15) / 16, (a_cols + 15) / 16);
    matmul_tn_kernel<<<blocks, threads>>>(a, d, output, reduction_rows, a_cols, d_cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_nt(const float* d, const float* weights, float* output, int rows,
                      int d_cols, int output_cols) {
    const dim3 threads(16, 16);
    const dim3 blocks((output_cols + 15) / 16, (rows + 15) / 16);
    matmul_nt_kernel<<<blocks, threads>>>(d, weights, output, rows, d_cols, output_cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_column_sum(const float* values, float* output, int rows, int cols) {
    constexpr int threads = 128;
    column_sum_kernel<<<(cols + threads - 1) / threads, threads>>>(values, output, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

void launch_relu_backward(float* gradient, const float* activations, int count) {
    constexpr int threads = 256;
    relu_backward_kernel<<<(count + threads - 1) / threads, threads>>>(gradient, activations,
                                                                      count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_sgd_update(float* parameter, const float* gradient, float learning_rate, int count) {
    constexpr int threads = 256;
    sgd_update_kernel<<<(count + threads - 1) / threads, threads>>>(parameter, gradient,
                                                                   learning_rate, count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_accuracy(const float* probabilities, const int* labels, int rows, int cols,
                     int* correct) {
    constexpr int threads = 256;
    accuracy_kernel<<<(rows + threads - 1) / threads, threads>>>(probabilities, labels, rows,
                                                                 cols, correct);
    CUDA_CHECK(cudaGetLastError());
}
