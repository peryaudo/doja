#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>

// Helper function declarations
size_t get_total_size(const std::vector<size_t>& shape);

// CUDA kernel declarations
extern "C" {
    void cuda_add(const float* a, const float* b, float* out, size_t size);
    void cuda_mul(const float* a, const float* b, float* out, size_t size);
    void cuda_matmul(const float* a, const float* b, float* out, int m, int n, int k);
    void cuda_relu(const float* input, float* output, size_t size);
    void cuda_relu_grad(const float* input, const float* grad, float* output, size_t size);
    void cuda_softmax(const float* input, float* output, int batch_size, int num_classes);
    void cuda_exp(const float* input, float* output, size_t size);
    void cuda_log(const float* input, float* output, size_t size);
    void cuda_div(const float* input, float scalar, float* output, size_t size);
    void cuda_sub(const float* a, const float* b, float* output, size_t size);
    void cuda_onehot(const int* indices, size_t num_classes, float* output, size_t batch_size);
    void cuda_sum(const float* input, int axis, bool keepdims, float* output, const size_t* shape, int num_dims);
    void cuda_max(const float* input, int axis, bool keepdims, float* output, const size_t* shape, int num_dims);
    void cuda_argmax(const float* input, int axis, int* output, const size_t* shape, int num_dims);
    void cuda_broadcast(const float* input, const size_t* input_shape, float* output, const size_t* target_shape, int num_dims);
    void cuda_pow(const float* input, float power, float* output, size_t size);
    void cuda_transpose(const float* input, float* output, int m, int n);
} 