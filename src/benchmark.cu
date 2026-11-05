#include "benchmark.h"

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "data.h"

namespace {

void cpu_train_step(std::vector<float>& W1, std::vector<float>& b1, std::vector<float>& W2,
                    std::vector<float>& b2, const float* features, const int* labels, int batch,
                    int in_dim, int hidden, int out_dim, float learning_rate) {
    std::vector<float> H(static_cast<size_t>(batch) * hidden);
    std::vector<float> dZ(static_cast<size_t>(batch) * out_dim);
    std::vector<float> dH(static_cast<size_t>(batch) * hidden);
    std::vector<float> dW1(W1.size(), 0.0f);
    std::vector<float> db1(hidden, 0.0f);
    std::vector<float> dW2(W2.size(), 0.0f);
    std::vector<float> db2(out_dim, 0.0f);
    std::vector<float> logits(out_dim);

    for (int row = 0; row < batch; ++row) {
        for (int h = 0; h < hidden; ++h) {
            float activation = b1[h];
            for (int input = 0; input < in_dim; ++input) {
                activation += features[row * in_dim + input] * W1[input * hidden + h];
            }
            H[static_cast<size_t>(row) * hidden + h] =
                activation > 0.0f ? activation : 0.0f;
        }

        float maximum = -FLT_MAX;
        for (int output = 0; output < out_dim; ++output) {
            float logit = b2[output];
            for (int h = 0; h < hidden; ++h) {
                logit += H[static_cast<size_t>(row) * hidden + h] *
                         W2[h * out_dim + output];
            }
            logits[output] = logit;
            maximum = std::fmax(maximum, logit);
        }

        float sum = 0.0f;
        for (float logit : logits) {
            sum += std::exp(logit - maximum);
        }
        for (int output = 0; output < out_dim; ++output) {
            dZ[static_cast<size_t>(row) * out_dim + output] =
                (std::exp(logits[output] - maximum) / sum -
                 (output == labels[row] ? 1.0f : 0.0f)) /
                batch;
        }
    }

    for (int row = 0; row < batch; ++row) {
        for (int h = 0; h < hidden; ++h) {
            const float activation = H[static_cast<size_t>(row) * hidden + h];
            float hidden_gradient = 0.0f;
            for (int output = 0; output < out_dim; ++output) {
                const float output_gradient =
                    dZ[static_cast<size_t>(row) * out_dim + output];
                dW2[h * out_dim + output] += activation * output_gradient;
                hidden_gradient += output_gradient * W2[h * out_dim + output];
            }
            dH[static_cast<size_t>(row) * hidden + h] =
                activation > 0.0f ? hidden_gradient : 0.0f;
        }
    }

    for (int row = 0; row < batch; ++row) {
        for (int output = 0; output < out_dim; ++output) {
            db2[output] += dZ[static_cast<size_t>(row) * out_dim + output];
        }
        for (int h = 0; h < hidden; ++h) {
            const float gradient = dH[static_cast<size_t>(row) * hidden + h];
            db1[h] += gradient;
            for (int input = 0; input < in_dim; ++input) {
                dW1[input * hidden + h] += features[row * in_dim + input] * gradient;
            }
        }
    }

    for (size_t index = 0; index < W1.size(); ++index) {
        W1[index] -= learning_rate * dW1[index];
    }
    for (int index = 0; index < hidden; ++index) {
        b1[index] -= learning_rate * db1[index];
    }
    for (size_t index = 0; index < W2.size(); ++index) {
        W2[index] -= learning_rate * dW2[index];
    }
    for (int index = 0; index < out_dim; ++index) {
        b2[index] -= learning_rate * db2[index];
    }
}

void time_both(int points_per_class, int gpu_repetitions, int cpu_repetitions) {
    constexpr int classes = 3;
    constexpr int in_dim = 2;
    constexpr int hidden = 64;
    const int batch = points_per_class * classes;

    std::vector<float> features(static_cast<size_t>(batch) * in_dim);
    std::vector<int> labels(batch);
    make_spiral(features.data(), labels.data(), points_per_class, classes, 0.2f, 5);
    float* device_features = device_upload(features.data(), batch * in_dim);
    int* device_labels = device_upload(labels.data(), batch);

    MLP gpu_model = init_mlp(in_dim, hidden, classes, 7);
    Workspace workspace = alloc_workspace(gpu_model, batch);
    const float gpu_ms = time_train_step_ms(gpu_model, workspace, device_features, device_labels,
                                             batch, 1.0f, gpu_repetitions);

    MLP initial_model = init_mlp(in_dim, hidden, classes, 7);
    std::vector<float> W1(in_dim * hidden);
    std::vector<float> b1(hidden);
    std::vector<float> W2(hidden * classes);
    std::vector<float> b2(classes);
    device_download(W1.data(), initial_model.W1, static_cast<int>(W1.size()));
    device_download(b1.data(), initial_model.b1, static_cast<int>(b1.size()));
    device_download(W2.data(), initial_model.W2, static_cast<int>(W2.size()));
    device_download(b2.data(), initial_model.b2, static_cast<int>(b2.size()));

    const auto start = std::chrono::high_resolution_clock::now();
    for (int repetition = 0; repetition < cpu_repetitions; ++repetition) {
        cpu_train_step(W1, b1, W2, b2, features.data(), labels.data(), batch, in_dim, hidden,
                       classes, 1.0f);
    }
    const auto stop = std::chrono::high_resolution_clock::now();
    const float cpu_ms =
        std::chrono::duration<float, std::milli>(stop - start).count() / cpu_repetitions;
    const bool gpu_is_faster = gpu_ms < cpu_ms;
    const float ratio = gpu_is_faster ? cpu_ms / gpu_ms : gpu_ms / cpu_ms;

    std::printf("  %8d points   GPU %8.3f ms/step   CPU %9.3f ms/step   GPU is %6.1fx %s\n",
                batch, gpu_ms, cpu_ms, ratio, gpu_is_faster ? "faster" : "slower");

    CUDA_CHECK(cudaFree(device_features));
    CUDA_CHECK(cudaFree(device_labels));
    free_workspace(workspace);
    free_mlp(gpu_model);
    free_mlp(initial_model);
}

}  // namespace

float time_train_step_ms(MLP& model, Workspace& workspace, const float* features,
                         const int* labels, int batch, float learning_rate, int repetitions) {
    train_step(model, workspace, features, labels, batch, learning_rate);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        train_step(model, workspace, features, labels, batch, learning_rate);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return elapsed_ms / repetitions;
}

void benchmark_training_steps() {
    std::printf("\nOne training step, GPU vs CPU (2 -> 64 -> 3, full batch):\n");
    time_both(200, 200, 20);
    time_both(4000, 50, 3);
    time_both(40000, 20, 1);
    std::printf(
        "\nThe CPU column is one core running the same loops. At large batches most of the "
        "GPU's time goes to the four\n"
        "kernels that sum over the batch (dW1, db1, dW2, db2): one thread per output means "
        "only 3 to 192 threads, each\n"
        "walking every row while the rest of the GPU idles. Splitting those sums across many "
        "blocks (a reduction like\n"
        "your cross-entropy kernel), fusing kernels and tiling the matrix products are what "
        "CUDA Intermediate covers.\n");
}
