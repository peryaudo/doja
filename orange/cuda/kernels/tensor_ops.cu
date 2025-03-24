#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Helper function to get the number of elements in a tensor
__device__ __host__ size_t get_num_elements(const int* shape, int ndim) {
    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}

// CUDA kernel for element-wise addition
__global__ void add_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for element-wise multiplication
__global__ void mul_kernel(const float* a, const float* b, float* out, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* a, const float* b, float* out,
                            int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        out[row * n + col] = sum;
    }
}

// CUDA kernel for ReLU
__global__ void relu_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

// CUDA kernel for ReLU gradient
__global__ void relu_grad_kernel(const float* input, const float* grad, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? grad[idx] : 0;
    }
}

// CUDA kernel for softmax
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    if (thread_idx < num_classes) {
        // Find max in the row
        float max_val = input[batch_idx * num_classes];
        for (int i = 1; i < num_classes; i++) {
            max_val = max(max_val, input[batch_idx * num_classes + i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < num_classes; i++) {
            float exp_val = __expf(input[batch_idx * num_classes + i] - max_val);
            output[batch_idx * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[batch_idx * num_classes + i] /= sum;
        }
    }
}

// Additional CUDA kernels
__global__ void exp_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

__global__ void log_kernel(const float* input, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

__global__ void div_kernel(const float* input, float scalar, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / scalar;
    }
}

__global__ void sub_kernel(const float* a, const float* b, float* output, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] - b[idx];
    }
}

__global__ void onehot_kernel(const int* indices, size_t num_classes, float* output, size_t batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int class_idx = indices[idx];
        for (size_t i = 0; i < num_classes; i++) {
            output[idx * num_classes + i] = (i == class_idx) ? 1.0f : 0.0f;
        }
    }
}

__global__ void sum_kernel(const float* input, int axis, bool keepdims, float* output, const size_t* shape, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < shape[axis]) {
        float sum = 0.0f;
        size_t stride = 1;
        for (int i = axis + 1; i < ndim; i++) {
            stride *= shape[i];
        }
        for (size_t i = 0; i < shape[axis]; i++) {
            sum += input[idx * stride + i];
        }
        output[idx] = sum;
    }
}

__global__ void max_kernel(const float* input, int axis, bool keepdims, float* output, const size_t* shape, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < shape[axis]) {
        float max_val = -INFINITY;
        size_t stride = 1;
        for (int i = axis + 1; i < ndim; i++) {
            stride *= shape[i];
        }
        for (size_t i = 0; i < shape[axis]; i++) {
            max_val = max(max_val, input[idx * stride + i]);
        }
        output[idx] = max_val;
    }
}

__global__ void argmax_kernel(const float* input, int axis, int* output, const size_t* shape, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < shape[axis]) {
        float max_val = -INFINITY;
        int max_idx = 0;
        size_t stride = 1;
        for (int i = axis + 1; i < ndim; i++) {
            stride *= shape[i];
        }
        for (size_t i = 0; i < shape[axis]; i++) {
            float val = input[idx * stride + i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        output[idx] = max_idx;
    }
}

__global__ void broadcast_kernel(const float* input, float* output, const size_t* input_shape, const size_t* output_shape, int ndim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_shape[0]) {
        size_t input_idx = 0;
        size_t output_idx = idx;
        for (int i = ndim - 1; i >= 0; i--) {
            size_t input_dim = input_shape[i];
            size_t output_dim = output_shape[i];
            size_t input_pos = output_idx % output_dim;
            input_idx += (input_pos % input_dim) * (i == 0 ? 1 : input_shape[i-1]);
            output_idx /= output_dim;
        }
        output[idx] = input[input_idx];
    }
}

// C++ wrapper functions for CUDA operations
extern "C" {

void cuda_add(const float* a, const float* b, float* out, size_t size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    add_kernel<<<num_blocks, block_size>>>(a, b, out, size);
}

void cuda_mul(const float* a, const float* b, float* out, size_t size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mul_kernel<<<num_blocks, block_size>>>(a, b, out, size);
}

void cuda_matmul(const float* a, const float* b, float* out, int m, int n, int k) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((n + block_dim.x - 1) / block_dim.x, (m + block_dim.y - 1) / block_dim.y);
    matmul_kernel<<<grid_dim, block_dim>>>(a, b, out, m, n, k);
}

void cuda_relu(const float* input, float* output, size_t size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    relu_kernel<<<num_blocks, block_size>>>(input, output, size);
}

void cuda_relu_grad(const float* input, const float* grad, float* output, size_t size) {
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    relu_grad_kernel<<<num_blocks, block_size>>>(input, grad, output, size);
}

void cuda_softmax(const float* input, float* output, int batch_size, int num_classes) {
    dim3 block_dim(num_classes);
    dim3 grid_dim(batch_size);
    softmax_kernel<<<grid_dim, block_dim>>>(input, output, batch_size, num_classes);
}

void cuda_exp(const float* input, float* output, const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    exp_kernel<<<num_blocks, block_size>>>(input, output, size);
}

void cuda_log(const float* input, float* output, const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    log_kernel<<<num_blocks, block_size>>>(input, output, size);
}

void cuda_div(const float* input, float scalar, float* output, const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    div_kernel<<<num_blocks, block_size>>>(input, scalar, output, size);
}

void cuda_sub(const float* a, const float* b, float* output, const std::vector<size_t>& shape) {
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sub_kernel<<<num_blocks, block_size>>>(a, b, output, size);
}

void cuda_onehot(const int* indices, size_t num_classes, float* output, const std::vector<size_t>& shape) {
    size_t batch_size = 1;
    for (size_t dim : shape) {
        batch_size *= dim;
    }
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    onehot_kernel<<<num_blocks, block_size>>>(indices, num_classes, output, batch_size);
}

void cuda_sum(const float* input, int axis, bool keepdims, float* output, const std::vector<size_t>& shape) {
    size_t size = shape[axis];
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    sum_kernel<<<num_blocks, block_size>>>(input, axis, keepdims, output, shape.data(), shape.size());
}

void cuda_max(const float* input, int axis, bool keepdims, float* output, const std::vector<size_t>& shape) {
    size_t size = shape[axis];
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    max_kernel<<<num_blocks, block_size>>>(input, axis, keepdims, output, shape.data(), shape.size());
}

void cuda_argmax(const float* input, int axis, int* output, const std::vector<size_t>& shape) {
    size_t size = shape[axis];
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    argmax_kernel<<<num_blocks, block_size>>>(input, axis, output, shape.data(), shape.size());
}

void cuda_broadcast(const float* input, const std::vector<size_t>& target_shape, float* output, const std::vector<size_t>& input_shape) {
    size_t size = 1;
    for (size_t dim : target_shape) {
        size *= dim;
    }
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    broadcast_kernel<<<num_blocks, block_size>>>(input, output, input_shape.data(), target_shape.data(), input_shape.size());
}

} // extern "C" 