"""
Train a Neural Network in CUDA from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

// scaffold.cu - train the two-layer MLP on the GPU, evaluate it, draw its decision regions,
// and time a training step against the same computation on the CPU.
// Your solutions from every step are concatenated above this file; main only drives them.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// One full-batch training step on the CPU: the same math as forward_mlp + backward_mlp + SGD.
// Used only to time the CPU against the GPU.
static void cpu_train_step(std::vector<float>& W1, std::vector<float>& b1, std::vector<float>& W2, std::vector<float>& b2,
                           const float* X, const int* y, int B, int I, int Hd, int O, float lr) {
    std::vector<float> H((size_t)B * Hd), dZ((size_t)B * O), dH((size_t)B * Hd);
    std::vector<float> gW1(W1.size(), 0.0f), gb1(Hd, 0.0f), gW2(W2.size(), 0.0f), gb2(O, 0.0f);
    for (int r = 0; r < B; ++r) {
        for (int h = 0; h < Hd; ++h) {
            float a = b1[h];
            for (int i = 0; i < I; ++i) a += X[r * I + i] * W1[i * Hd + h];
            H[(size_t)r * Hd + h] = a > 0.0f ? a : 0.0f;
        }
        float z[16], m = -FLT_MAX, s = 0.0f;
        for (int o = 0; o < O; ++o) {
            float a = b2[o];
            for (int h = 0; h < Hd; ++h) a += H[(size_t)r * Hd + h] * W2[h * O + o];
            z[o] = a;
            m = fmaxf(m, a);
        }
        for (int o = 0; o < O; ++o) s += expf(z[o] - m);
        for (int o = 0; o < O; ++o) dZ[(size_t)r * O + o] = (expf(z[o] - m) / s - (o == y[r] ? 1.0f : 0.0f)) / B;
    }
    for (int r = 0; r < B; ++r)
        for (int h = 0; h < Hd; ++h) {
            float hv = H[(size_t)r * Hd + h], a = 0.0f;
            for (int o = 0; o < O; ++o) {
                gW2[h * O + o] += hv * dZ[(size_t)r * O + o];
                a += dZ[(size_t)r * O + o] * W2[h * O + o];
            }
            dH[(size_t)r * Hd + h] = hv > 0.0f ? a : 0.0f;
        }
    for (int r = 0; r < B; ++r) {
        for (int o = 0; o < O; ++o) gb2[o] += dZ[(size_t)r * O + o];
        for (int h = 0; h < Hd; ++h) {
            float g = dH[(size_t)r * Hd + h];
            gb1[h] += g;
            for (int i = 0; i < I; ++i) gW1[i * Hd + h] += X[r * I + i] * g;
        }
    }
    for (size_t k = 0; k < W1.size(); ++k) W1[k] -= lr * gW1[k];
    for (int k = 0; k < Hd; ++k) b1[k] -= lr * gb1[k];
    for (size_t k = 0; k < W2.size(); ++k) W2[k] -= lr * gW2[k];
    for (int k = 0; k < O; ++k) b2[k] -= lr * gb2[k];
}

// Predict the class at every cell of a 64 x 24 grid over [-1.1, 1.1]^2 and print the regions,
// with the training points of each class drawn on top as A, B, C.
static void draw_regions(const MLP& m, const std::vector<float>& X, const std::vector<int>& y) {
    const int GW = 64, GH = 24, G = GW * GH;
    std::vector<float> grid((size_t)G * 2);
    for (int gy = 0; gy < GH; ++gy)
        for (int gx = 0; gx < GW; ++gx) {
            int k = gy * GW + gx;
            grid[2 * k] = -1.1f + 2.2f * gx / (GW - 1);
            grid[2 * k + 1] = 1.1f - 2.2f * gy / (GH - 1);
        }
    Workspace wg = alloc_workspace(m, G);
    float* dgrid = device_upload(grid.data(), G * 2);
    forward_mlp(m, dgrid, G, wg);
    std::vector<float> P((size_t)G * m.out_dim);
    device_download(P.data(), wg.P, G * m.out_dim);
    std::vector<char> canvas(G);
    const char region[3] = {'.', '-', ' '};
    for (int k = 0; k < G; ++k) {
        int best = 0;
        for (int j = 1; j < m.out_dim; ++j) if (P[(size_t)k * m.out_dim + j] > P[(size_t)k * m.out_dim + best]) best = j;
        canvas[k] = region[best % 3];
    }
    for (size_t i = 0; i < y.size(); i += 4) {
        int gx = (int)lroundf((X[2 * i] + 1.1f) / 2.2f * (GW - 1));
        int gy = (int)lroundf((1.1f - X[2 * i + 1]) / 2.2f * (GH - 1));
        if (gx >= 0 && gx < GW && gy >= 0 && gy < GH) canvas[gy * GW + gx] = (char)('A' + y[i]);
    }
    printf("\nDecision regions ('.' = class 0, '-' = class 1, ' ' = class 2; every 4th training point as A/B/C):\n");
    for (int gy = 0; gy < GH; ++gy) {
        printf("  |");
        for (int gx = 0; gx < GW; ++gx) putchar(canvas[gy * GW + gx]);
        printf("|\n");
    }
    cudaFree(dgrid);
    free_workspace(wg);
}

static void time_both(int points_per_class, int reps_gpu, int reps_cpu) {
    const int B = points_per_class * 3;
    std::vector<float> X((size_t)B * 2);
    std::vector<int> y(B);
    make_spiral(X.data(), y.data(), points_per_class, 3, 0.2f, 5);
    float* dX = device_upload(X.data(), B * 2);
    int* dy = device_upload(y.data(), B);
    MLP g = init_mlp(2, 64, 3, 7);
    Workspace w = alloc_workspace(g, B);
    float gpu_ms = time_train_step_ms(g, w, dX, dy, B, 1.0f, reps_gpu);

    MLP c = init_mlp(2, 64, 3, 7);
    std::vector<float> W1(2 * 64), b1(64), W2(64 * 3), b2(3);
    device_download(W1.data(), c.W1, 128);
    device_download(b1.data(), c.b1, 64);
    device_download(W2.data(), c.W2, 192);
    device_download(b2.data(), c.b2, 3);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps_cpu; ++i) cpu_train_step(W1, b1, W2, b2, X.data(), y.data(), B, 2, 64, 3, 1.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / reps_cpu;
    printf("  %8d points   GPU %8.3f ms/step   CPU %9.3f ms/step   GPU is %6.1fx %s\n", B, gpu_ms, cpu_ms,
           gpu_ms < cpu_ms ? cpu_ms / gpu_ms : gpu_ms / cpu_ms, gpu_ms < cpu_ms ? "faster" : "slower");
    cudaFree(dX); cudaFree(dy);
    free_workspace(w); free_mlp(g); free_mlp(c);
}

int main() {
    cudaDeviceProp prop;
    if (report_cuda_error(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 1;
    printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    // ---- 1. Data: 600 training points and 300 held-out points from the same three spirals ----
    const int TRAIN = 600, TEST = 300;
    std::vector<float> X(TRAIN * 2), Xt(TEST * 2);
    std::vector<int> y(TRAIN), yt(TEST);
    make_spiral(X.data(), y.data(), 200, 3, 0.2f, 1);
    make_spiral(Xt.data(), yt.data(), 100, 3, 0.2f, 2);
    float* dX = device_upload(X.data(), TRAIN * 2);
    int* dy = device_upload(y.data(), TRAIN);
    float* dXt = device_upload(Xt.data(), TEST * 2);
    int* dyt = device_upload(yt.data(), TEST);

    // ---- 2. Train a 2 -> 64 -> 3 MLP with full-batch gradient descent ----
    MLP m = init_mlp(2, 64, 3, 7);
    Workspace w = alloc_workspace(m, TRAIN);
    forward_mlp(m, dX, TRAIN, w);
    printf("\nBefore training: train accuracy %.3f (chance is 0.333)\n", accuracy(w.P, dy, TRAIN, 3));
    printf("\nTraining 2 -> 64 -> 3, lr 1.0, 2000 full-batch epochs:\n");
    float final_loss = train_mlp(m, w, dX, dy, TRAIN, 1.0f, 2000, 250);

    // ---- 3. Evaluate ----
    forward_mlp(m, dX, TRAIN, w);
    float train_acc = accuracy(w.P, dy, TRAIN, 3);
    forward_mlp(m, dXt, TEST, w);
    float test_acc = accuracy(w.P, dyt, TEST, 3);
    float test_loss = cross_entropy_loss(w.P, dyt, TEST, 3);
    printf("\nFinal: train loss %.4f, train accuracy %.3f | held-out loss %.4f, held-out accuracy %.3f\n",
           final_loss, train_acc, test_loss, test_acc);
    draw_regions(m, X, y);

    // ---- 4. What a step costs: GPU (your kernels) vs CPU (the same math, one core) ----
    printf("\nOne training step, GPU vs CPU (2 -> 64 -> 3, full batch):\n");
    time_both(200, 200, 20);
    time_both(4000, 50, 3);
    time_both(40000, 20, 1);
    printf("\nThe CPU column is one core running the same loops. At large batches most of the GPU's time goes to the four\n");
    printf("kernels that sum over the batch (dW1, db1, dW2, db2): one thread per output means only 3 to 192 threads, each\n");
    printf("walking every row while the rest of the GPU idles. Splitting those sums across many blocks (a reduction like\n");
    printf("your cross-entropy kernel), fusing kernels and tiling the matrix products are what CUDA Intermediate covers.\n");

    cudaFree(dX); cudaFree(dy); cudaFree(dXt); cudaFree(dyt);
    free_workspace(w);
    free_mlp(m);
    return 0;
}

