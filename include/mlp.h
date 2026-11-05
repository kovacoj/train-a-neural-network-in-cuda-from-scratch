#pragma once

struct MLP {
    int in_dim = 0;
    int hidden = 0;
    int out_dim = 0;
    float* W1 = nullptr;
    float* b1 = nullptr;
    float* W2 = nullptr;
    float* b2 = nullptr;

    MLP() = default;
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;
    MLP(MLP&& other) noexcept;
    MLP& operator=(MLP&&) = delete;
};

struct Workspace {
    int batch = 0;
    int in_dim = 0;
    int hidden = 0;
    int out_dim = 0;
    float* H = nullptr;
    float* P = nullptr;
    float* dZ2 = nullptr;
    float* dH = nullptr;
    float* dW1 = nullptr;
    float* db1 = nullptr;
    float* dW2 = nullptr;
    float* db2 = nullptr;
    float* loss_total = nullptr;
    int* correct = nullptr;

    Workspace() = default;
    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;
    Workspace(Workspace&& other) noexcept;
    Workspace& operator=(Workspace&&) = delete;
};

float* device_upload(const float* host, int count);
int* device_upload(const int* host, int count);
void device_download(float* host, const float* device, int count);

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed);
void free_mlp(MLP& model);
Workspace alloc_workspace(const MLP& model, int batch);
void free_workspace(Workspace& workspace);

void forward_mlp(const MLP& model, const float* features, int batch, Workspace& workspace);
void backward_mlp(const MLP& model, const float* features, const int* labels, int batch,
                  Workspace& workspace);
float cross_entropy_loss(Workspace& workspace, const int* labels, int rows);
float accuracy(Workspace& workspace, const int* labels, int rows);
float train_step(MLP& model, Workspace& workspace, const float* features, const int* labels,
                 int batch, float learning_rate);
float train_mlp(MLP& model, Workspace& workspace, const float* features, const int* labels,
                int batch, float learning_rate, int epochs, int print_every);
