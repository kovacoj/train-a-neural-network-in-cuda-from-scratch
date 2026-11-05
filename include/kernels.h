#pragma once

void launch_matmul(const float* a, const float* b, float* output, int rows, int cols,
                   int inner);
void launch_add_bias(float* values, const float* bias, int rows, int cols);
void launch_relu(float* values, int count);
void launch_softmax_rows(const float* logits, float* probabilities, int rows, int cols);
void launch_cross_entropy(const float* probabilities, const int* labels, int rows, int cols,
                          float* total);
void launch_softmax_ce_grad(const float* probabilities, const int* labels, float* gradient,
                            int rows, int cols);
void launch_matmul_tn(const float* a, const float* d, float* output, int reduction_rows,
                      int a_cols, int d_cols);
void launch_matmul_nt(const float* d, const float* weights, float* output, int rows,
                      int d_cols, int output_cols);
void launch_column_sum(const float* values, float* output, int rows, int cols);
void launch_relu_backward(float* gradient, const float* activations, int count);
void launch_sgd_update(float* parameter, const float* gradient, float learning_rate, int count);
void launch_accuracy(const float* probabilities, const int* labels, int rows, int cols,
                     int* correct);
