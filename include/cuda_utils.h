#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

inline void check_cuda(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", file, line, expression,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)
