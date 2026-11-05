#include "mlp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "data.h"
#include "kernels.h"

namespace {

template <typename T>
void allocate_device(T** pointer, int count) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(pointer), sizeof(T) * count));
}

template <typename T>
void allocate_zeroed_device(T** pointer, int count) {
    allocate_device(pointer, count);
    CUDA_CHECK(cudaMemset(*pointer, 0, sizeof(T) * count));
}

void require_compatible(const Workspace& workspace, const MLP& model, int batch) {
    if (batch <= workspace.batch && workspace.in_dim == model.in_dim &&
        workspace.hidden == model.hidden && workspace.out_dim == model.out_dim) {
        return;
    }

    std::fprintf(stderr,
                 "Workspace [%d, %d, %d, %d] cannot serve model batch [%d, %d, %d, %d]\n",
                 workspace.batch, workspace.in_dim, workspace.hidden, workspace.out_dim, batch,
                 model.in_dim, model.hidden, model.out_dim);
    std::exit(EXIT_FAILURE);
}

}  // namespace

MLP::MLP(MLP&& other) noexcept
    : in_dim(other.in_dim),
      hidden(other.hidden),
      out_dim(other.out_dim),
      W1(other.W1),
      b1(other.b1),
      W2(other.W2),
      b2(other.b2) {
    other.W1 = nullptr;
    other.b1 = nullptr;
    other.W2 = nullptr;
    other.b2 = nullptr;
}

Workspace::Workspace(Workspace&& other) noexcept
    : batch(other.batch),
      in_dim(other.in_dim),
      hidden(other.hidden),
      out_dim(other.out_dim),
      H(other.H),
      P(other.P),
      dZ2(other.dZ2),
      dH(other.dH),
      dW1(other.dW1),
      db1(other.db1),
      dW2(other.dW2),
      db2(other.db2),
      loss_total(other.loss_total),
      correct(other.correct) {
    other.H = nullptr;
    other.P = nullptr;
    other.dZ2 = nullptr;
    other.dH = nullptr;
    other.dW1 = nullptr;
    other.db1 = nullptr;
    other.dW2 = nullptr;
    other.db2 = nullptr;
    other.loss_total = nullptr;
    other.correct = nullptr;
}

float* device_upload(const float* host, int count) {
    float* device = nullptr;
    allocate_device(&device, count);
    CUDA_CHECK(cudaMemcpy(device, host, sizeof(float) * count, cudaMemcpyHostToDevice));
    return device;
}

int* device_upload(const int* host, int count) {
    int* device = nullptr;
    allocate_device(&device, count);
    CUDA_CHECK(cudaMemcpy(device, host, sizeof(int) * count, cudaMemcpyHostToDevice));
    return device;
}

void device_download(float* host, const float* device, int count) {
    CUDA_CHECK(cudaMemcpy(host, device, sizeof(float) * count, cudaMemcpyDeviceToHost));
}

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed) {
    MLP model;
    model.in_dim = in_dim;
    model.hidden = hidden;
    model.out_dim = out_dim;
    std::vector<float> W1(in_dim * hidden);
    std::vector<float> b1(hidden, 0.0f);
    std::vector<float> W2(hidden * out_dim);
    std::vector<float> b2(out_dim, 0.0f);

    unsigned state = seed;
    const float first_scale = std::sqrt(6.0f / in_dim);
    for (float& weight : W1) {
        weight = (2.0f * lcg_uniform(&state) - 1.0f) * first_scale;
    }

    const float second_scale = std::sqrt(6.0f / hidden);
    for (float& weight : W2) {
        weight = (2.0f * lcg_uniform(&state) - 1.0f) * second_scale;
    }

    model.W1 = device_upload(W1.data(), static_cast<int>(W1.size()));
    model.b1 = device_upload(b1.data(), static_cast<int>(b1.size()));
    model.W2 = device_upload(W2.data(), static_cast<int>(W2.size()));
    model.b2 = device_upload(b2.data(), static_cast<int>(b2.size()));
    return model;
}

void free_mlp(MLP& model) {
    CUDA_CHECK(cudaFree(model.W1));
    CUDA_CHECK(cudaFree(model.b1));
    CUDA_CHECK(cudaFree(model.W2));
    CUDA_CHECK(cudaFree(model.b2));
    model.W1 = nullptr;
    model.b1 = nullptr;
    model.W2 = nullptr;
    model.b2 = nullptr;
}

Workspace alloc_workspace(const MLP& model, int batch) {
    Workspace workspace;
    workspace.batch = batch;
    workspace.in_dim = model.in_dim;
    workspace.hidden = model.hidden;
    workspace.out_dim = model.out_dim;
    allocate_zeroed_device(&workspace.H, batch * model.hidden);
    allocate_zeroed_device(&workspace.P, batch * model.out_dim);
    allocate_zeroed_device(&workspace.dZ2, batch * model.out_dim);
    allocate_zeroed_device(&workspace.dH, batch * model.hidden);
    allocate_zeroed_device(&workspace.dW1, model.in_dim * model.hidden);
    allocate_zeroed_device(&workspace.db1, model.hidden);
    allocate_zeroed_device(&workspace.dW2, model.hidden * model.out_dim);
    allocate_zeroed_device(&workspace.db2, model.out_dim);
    allocate_zeroed_device(&workspace.loss_total, 1);
    allocate_zeroed_device(&workspace.correct, 1);
    return workspace;
}

void free_workspace(Workspace& workspace) {
    CUDA_CHECK(cudaFree(workspace.H));
    CUDA_CHECK(cudaFree(workspace.P));
    CUDA_CHECK(cudaFree(workspace.dZ2));
    CUDA_CHECK(cudaFree(workspace.dH));
    CUDA_CHECK(cudaFree(workspace.dW1));
    CUDA_CHECK(cudaFree(workspace.db1));
    CUDA_CHECK(cudaFree(workspace.dW2));
    CUDA_CHECK(cudaFree(workspace.db2));
    CUDA_CHECK(cudaFree(workspace.loss_total));
    CUDA_CHECK(cudaFree(workspace.correct));

    workspace.H = nullptr;
    workspace.P = nullptr;
    workspace.dZ2 = nullptr;
    workspace.dH = nullptr;
    workspace.dW1 = nullptr;
    workspace.db1 = nullptr;
    workspace.dW2 = nullptr;
    workspace.db2 = nullptr;
    workspace.loss_total = nullptr;
    workspace.correct = nullptr;
}

void forward_mlp(const MLP& model, const float* features, int batch, Workspace& workspace) {
    require_compatible(workspace, model, batch);
    launch_matmul(features, model.W1, workspace.H, batch, model.hidden, model.in_dim);
    launch_add_bias(workspace.H, model.b1, batch, model.hidden);
    launch_relu(workspace.H, batch * model.hidden);

    launch_matmul(workspace.H, model.W2, workspace.P, batch, model.out_dim, model.hidden);
    launch_add_bias(workspace.P, model.b2, batch, model.out_dim);
    launch_softmax_rows(workspace.P, workspace.P, batch, model.out_dim);
}

void backward_mlp(const MLP& model, const float* features, const int* labels, int batch,
                  Workspace& workspace) {
    require_compatible(workspace, model, batch);
    launch_softmax_ce_grad(workspace.P, labels, workspace.dZ2, batch, model.out_dim);
    launch_matmul_tn(workspace.H, workspace.dZ2, workspace.dW2, batch, model.hidden,
                     model.out_dim);
    launch_column_sum(workspace.dZ2, workspace.db2, batch, model.out_dim);

    launch_matmul_nt(workspace.dZ2, model.W2, workspace.dH, batch, model.out_dim,
                     model.hidden);
    launch_relu_backward(workspace.dH, workspace.H, batch * model.hidden);
    launch_matmul_tn(features, workspace.dH, workspace.dW1, batch, model.in_dim,
                     model.hidden);
    launch_column_sum(workspace.dH, workspace.db1, batch, model.hidden);
}

float cross_entropy_loss(Workspace& workspace, const int* labels, int rows) {
    if (rows > workspace.batch) {
        std::fprintf(stderr, "Workspace capacity %d is smaller than batch %d\n", workspace.batch,
                     rows);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemset(workspace.loss_total, 0, sizeof(float)));
    launch_cross_entropy(workspace.P, labels, rows, workspace.out_dim, workspace.loss_total);

    float total = 0.0f;
    CUDA_CHECK(cudaMemcpy(&total, workspace.loss_total, sizeof(float), cudaMemcpyDeviceToHost));
    return total / rows;
}

float accuracy(Workspace& workspace, const int* labels, int rows) {
    if (rows > workspace.batch) {
        std::fprintf(stderr, "Workspace capacity %d is smaller than batch %d\n", workspace.batch,
                     rows);
        std::exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemset(workspace.correct, 0, sizeof(int)));
    launch_accuracy(workspace.P, labels, rows, workspace.out_dim, workspace.correct);

    int correct = 0;
    CUDA_CHECK(cudaMemcpy(&correct, workspace.correct, sizeof(int), cudaMemcpyDeviceToHost));
    return correct / static_cast<float>(rows);
}

float train_step(MLP& model, Workspace& workspace, const float* features, const int* labels,
                 int batch, float learning_rate) {
    forward_mlp(model, features, batch, workspace);
    const float loss = cross_entropy_loss(workspace, labels, batch);
    backward_mlp(model, features, labels, batch, workspace);

    launch_sgd_update(model.W1, workspace.dW1, learning_rate, model.in_dim * model.hidden);
    launch_sgd_update(model.b1, workspace.db1, learning_rate, model.hidden);
    launch_sgd_update(model.W2, workspace.dW2, learning_rate, model.hidden * model.out_dim);
    launch_sgd_update(model.b2, workspace.db2, learning_rate, model.out_dim);
    return loss;
}

float train_mlp(MLP& model, Workspace& workspace, const float* features, const int* labels,
                int batch, float learning_rate, int epochs, int print_every) {
    float loss = 0.0f;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        loss = train_step(model, workspace, features, labels, batch, learning_rate);
        if (print_every > 0 && (epoch == 1 || epoch % print_every == 0)) {
            std::printf("epoch %5d loss %.4f\n", epoch, loss);
        }
    }
    return loss;
}
