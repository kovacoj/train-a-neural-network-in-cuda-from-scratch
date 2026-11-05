#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "benchmark.h"
#include "cuda_utils.h"
#include "data.h"
#include "mlp.h"

namespace {

void draw_regions(const MLP& model, const std::vector<float>& features,
                  const std::vector<int>& labels) {
    constexpr int grid_width = 64;
    constexpr int grid_height = 24;
    constexpr int grid_size = grid_width * grid_height;
    std::vector<float> grid(grid_size * 2);

    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int index = y * grid_width + x;
            grid[2 * index] = -1.1f + 2.2f * x / (grid_width - 1);
            grid[2 * index + 1] = 1.1f - 2.2f * y / (grid_height - 1);
        }
    }

    Workspace workspace = alloc_workspace(model, grid_size);
    float* device_grid = device_upload(grid.data(), grid_size * 2);
    forward_mlp(model, device_grid, grid_size, workspace);

    std::vector<float> probabilities(grid_size * model.out_dim);
    device_download(probabilities.data(), workspace.P,
                    static_cast<int>(probabilities.size()));
    std::vector<char> canvas(grid_size);
    constexpr char regions[3] = {'.', '-', ' '};
    for (int index = 0; index < grid_size; ++index) {
        int predicted = 0;
        for (int output = 1; output < model.out_dim; ++output) {
            if (probabilities[index * model.out_dim + output] >
                probabilities[index * model.out_dim + predicted]) {
                predicted = output;
            }
        }
        canvas[index] = regions[predicted % 3];
    }

    for (size_t index = 0; index < labels.size(); index += 4) {
        const int x = static_cast<int>(
            std::lround((features[2 * index] + 1.1f) / 2.2f * (grid_width - 1)));
        const int y = static_cast<int>(
            std::lround((1.1f - features[2 * index + 1]) / 2.2f * (grid_height - 1)));
        if (x >= 0 && x < grid_width && y >= 0 && y < grid_height) {
            canvas[y * grid_width + x] = static_cast<char>('A' + labels[index]);
        }
    }

    std::printf(
        "\nDecision regions ('.' = class 0, '-' = class 1, ' ' = class 2; every 4th "
        "training point as A/B/C):\n");
    for (int y = 0; y < grid_height; ++y) {
        std::printf("  |");
        for (int x = 0; x < grid_width; ++x) {
            std::putchar(canvas[y * grid_width + x]);
        }
        std::printf("|\n");
    }

    CUDA_CHECK(cudaFree(device_grid));
    free_workspace(workspace);
}

}  // namespace

int main() {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("GPU: %s, %d SMs\n", properties.name, properties.multiProcessorCount);

    constexpr int train_size = 600;
    constexpr int test_size = 300;
    constexpr int classes = 3;
    std::vector<float> train_features(train_size * 2);
    std::vector<int> train_labels(train_size);
    std::vector<float> test_features(test_size * 2);
    std::vector<int> test_labels(test_size);
    make_spiral(train_features.data(), train_labels.data(), 200, classes, 0.2f, 1);
    make_spiral(test_features.data(), test_labels.data(), 100, classes, 0.2f, 2);

    float* device_train_features = device_upload(train_features.data(), train_size * 2);
    int* device_train_labels = device_upload(train_labels.data(), train_size);
    float* device_test_features = device_upload(test_features.data(), test_size * 2);
    int* device_test_labels = device_upload(test_labels.data(), test_size);

    MLP model = init_mlp(2, 64, classes, 7);
    Workspace workspace = alloc_workspace(model, train_size);
    forward_mlp(model, device_train_features, train_size, workspace);
    std::printf("\nBefore training: train accuracy %.3f (chance is 0.333)\n",
                accuracy(workspace, device_train_labels, train_size));

    std::printf("\nTraining 2 -> 64 -> 3, lr 1.0, 2000 full-batch epochs:\n");
    const float final_loss = train_mlp(model, workspace, device_train_features,
                                       device_train_labels, train_size, 1.0f, 2000, 250);

    forward_mlp(model, device_train_features, train_size, workspace);
    const float train_accuracy =
        accuracy(workspace, device_train_labels, train_size);
    forward_mlp(model, device_test_features, test_size, workspace);
    const float test_accuracy = accuracy(workspace, device_test_labels, test_size);
    const float test_loss = cross_entropy_loss(workspace, device_test_labels, test_size);
    std::printf(
        "\nFinal: train loss %.4f, train accuracy %.3f | held-out loss %.4f, held-out "
        "accuracy %.3f\n",
        final_loss, train_accuracy, test_loss, test_accuracy);

    draw_regions(model, train_features, train_labels);
    benchmark_training_steps();

    CUDA_CHECK(cudaFree(device_train_features));
    CUDA_CHECK(cudaFree(device_train_labels));
    CUDA_CHECK(cudaFree(device_test_features));
    CUDA_CHECK(cudaFree(device_test_labels));
    free_workspace(workspace);
    free_mlp(model);
    return 0;
}
