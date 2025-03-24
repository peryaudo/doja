#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../kernels/tensor_ops.h"

namespace py = pybind11;

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error));
    }
}

// RAII wrapper for CUDA memory
class CUDATensor {
public:
    CUDATensor(const std::vector<size_t>& shape) : shape_(shape) {
        size_t size = get_total_size(shape);
        check_cuda_error(cudaMalloc(&data_, size * sizeof(float)));
    }

    CUDATensor(const std::vector<size_t>& shape, const float* host_data) : shape_(shape) {
        size_t size = get_total_size(shape);
        check_cuda_error(cudaMalloc(&data_, size * sizeof(float)));
        check_cuda_error(cudaMemcpy(data_, host_data, size * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~CUDATensor() {
        if (data_) {
            cudaFree(data_);
        }
    }

    // Delete copy constructor and assignment operator
    CUDATensor(const CUDATensor&) = delete;
    CUDATensor& operator=(const CUDATensor&) = delete;

    // Allow move constructor and assignment
    CUDATensor(CUDATensor&& other) noexcept : data_(other.data_), shape_(std::move(other.shape_)) {
        other.data_ = nullptr;
    }
    CUDATensor& operator=(CUDATensor&& other) noexcept {
        if (this != &other) {
            if (data_) {
                cudaFree(data_);
            }
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            other.data_ = nullptr;
        }
        return *this;
    }

    float* data() const { return data_; }
    const std::vector<size_t>& shape() const { return shape_; }

    // Convert shape to Python tuple
    py::tuple shape_tuple() const {
        py::tuple result(shape_.size());
        for (size_t i = 0; i < shape_.size(); i++) {
            result[i] = shape_[i];
        }
        return result;
    }

    // Copy data to host
    py::array_t<float> to_cpu() const {
        size_t size = get_total_size(shape_);
        py::array_t<float> result(shape_);
        py::buffer_info result_buf = result.request();
        float* result_ptr = (float*)result_buf.ptr;
        
        check_cuda_error(cudaMemcpy(result_ptr, data_, size * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }

private:
    float* data_;
    std::vector<size_t> shape_;
};

// Forward declarations of wrapper functions
CUDATensor cuda_broadcast_wrapper(const CUDATensor& input, const std::vector<size_t>& target_shape);

// Function to create CUDA tensor from numpy array
CUDATensor create_cuda_tensor(py::array_t<float> input) {
    py::buffer_info input_buf = input.request();
    std::vector<size_t> shape(input_buf.shape.begin(), input_buf.shape.end());
    return CUDATensor(shape, (float*)input_buf.ptr);
}

// Helper function to check if shapes are broadcastable
bool is_broadcastable(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    if (shape1.size() < shape2.size()) {
        return is_broadcastable(shape2, shape1);
    }
    
    for (size_t i = 0; i < shape2.size(); i++) {
        size_t dim1 = shape1[shape1.size() - 1 - i];
        size_t dim2 = shape2[shape2.size() - 1 - i];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }
    return true;
}

// Helper function to get broadcasted shape
std::vector<size_t> get_broadcasted_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    std::vector<size_t> result;
    int i = shape1.size() - 1;
    int j = shape2.size() - 1;
    
    while (i >= 0 || j >= 0) {
        size_t dim1 = (i >= 0) ? shape1[i] : 1;
        size_t dim2 = (j >= 0) ? shape2[j] : 1;
        result.insert(result.begin(), std::max(dim1, dim2));
        i--;
        j--;
    }
    return result;
}

// CUDA tensor operations
CUDATensor cuda_add_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() == b.shape()) {
        CUDATensor out(a.shape());
        cuda_add(a.data(), b.data(), out.data(), get_total_size(a.shape()));
        return out;
    }
    
    if (!is_broadcastable(a.shape(), b.shape())) {
        throw std::runtime_error("Shapes are not broadcastable");
    }
    
    std::vector<size_t> out_shape = get_broadcasted_shape(a.shape(), b.shape());
    CUDATensor out(out_shape);
    
    // Broadcast smaller tensor to match larger tensor's shape
    if (a.shape() == out_shape) {
        CUDATensor b_broadcasted = cuda_broadcast_wrapper(b, out_shape);
        cuda_add(a.data(), b_broadcasted.data(), out.data(), get_total_size(out_shape));
    } else {
        CUDATensor a_broadcasted = cuda_broadcast_wrapper(a, out_shape);
        cuda_add(a_broadcasted.data(), b.data(), out.data(), get_total_size(out_shape));
    }
    return out;
}

CUDATensor cuda_mul_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() == b.shape()) {
        CUDATensor out(a.shape());
        cuda_mul(a.data(), b.data(), out.data(), get_total_size(a.shape()));
        return out;
    }
    
    if (!is_broadcastable(a.shape(), b.shape())) {
        throw std::runtime_error("Shapes are not broadcastable");
    }
    
    std::vector<size_t> out_shape = get_broadcasted_shape(a.shape(), b.shape());
    CUDATensor out(out_shape);
    
    // Broadcast smaller tensor to match larger tensor's shape
    if (a.shape() == out_shape) {
        CUDATensor b_broadcasted = cuda_broadcast_wrapper(b, out_shape);
        cuda_mul(a.data(), b_broadcasted.data(), out.data(), get_total_size(out_shape));
    } else {
        CUDATensor a_broadcasted = cuda_broadcast_wrapper(a, out_shape);
        cuda_mul(a_broadcasted.data(), b.data(), out.data(), get_total_size(out_shape));
    }
    return out;
}

CUDATensor cuda_matmul_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("Inputs must be 2D arrays");
    }
    
    size_t m = a.shape()[0];
    size_t k = a.shape()[1];
    size_t n = b.shape()[1];
    
    if (k != b.shape()[0]) {
        throw std::runtime_error("Matrix dimensions must match for multiplication");
    }
    
    CUDATensor out({m, n});
    cuda_matmul(a.data(), b.data(), out.data(), m, n, k);
    return out;
}

CUDATensor cuda_relu_wrapper(const CUDATensor& input) {
    CUDATensor out(input.shape());
    cuda_relu(input.data(), out.data(), get_total_size(input.shape()));
    return out;
}

CUDATensor cuda_softmax_wrapper(const CUDATensor& input) {
    if (input.shape().size() != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    
    CUDATensor out(input.shape());
    cuda_softmax(input.data(), out.data(), input.shape()[0], input.shape()[1]);
    return out;
}

CUDATensor cuda_exp_wrapper(const CUDATensor& input) {
    CUDATensor out(input.shape());
    cuda_exp(input.data(), out.data(), get_total_size(input.shape()));
    return out;
}

CUDATensor cuda_log_wrapper(const CUDATensor& input) {
    CUDATensor out(input.shape());
    cuda_log(input.data(), out.data(), get_total_size(input.shape()));
    return out;
}

CUDATensor cuda_div_wrapper(const CUDATensor& a, float b) {
    CUDATensor out(a.shape());
    cuda_div(a.data(), b, out.data(), get_total_size(a.shape()));
    return out;
}

CUDATensor cuda_sub_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() == b.shape()) {
        CUDATensor out(a.shape());
        cuda_sub(a.data(), b.data(), out.data(), get_total_size(a.shape()));
        return out;
    }
    
    if (!is_broadcastable(a.shape(), b.shape())) {
        throw std::runtime_error("Shapes are not broadcastable");
    }
    
    std::vector<size_t> out_shape = get_broadcasted_shape(a.shape(), b.shape());
    CUDATensor out(out_shape);
    
    // Broadcast smaller tensor to match larger tensor's shape
    if (a.shape() == out_shape) {
        CUDATensor b_broadcasted = cuda_broadcast_wrapper(b, out_shape);
        cuda_sub(a.data(), b_broadcasted.data(), out.data(), get_total_size(out_shape));
    } else {
        CUDATensor a_broadcasted = cuda_broadcast_wrapper(a, out_shape);
        cuda_sub(a_broadcasted.data(), b.data(), out.data(), get_total_size(out_shape));
    }
    return out;
}

CUDATensor cuda_onehot_wrapper(const CUDATensor& indices, size_t num_classes) {
    std::vector<size_t> shape = indices.shape();
    shape.push_back(num_classes);
    CUDATensor out(shape);
    
    // Convert float indices to int
    std::vector<int> int_indices(get_total_size(indices.shape()));
    py::array_t<float> cpu_indices = indices.to_cpu();
    float* indices_ptr = (float*)cpu_indices.request().ptr;
    for (size_t i = 0; i < int_indices.size(); i++) {
        int_indices[i] = static_cast<int>(indices_ptr[i]);
    }
    
    // Copy int indices to GPU
    int* d_indices;
    cudaMalloc(&d_indices, int_indices.size() * sizeof(int));
    cudaMemcpy(d_indices, int_indices.data(), int_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    cuda_onehot(d_indices, num_classes, out.data(), get_total_size(indices.shape()));
    
    cudaFree(d_indices);
    return out;
}

CUDATensor cuda_sum_wrapper(const CUDATensor& input, int axis, bool keepdims) {
    std::vector<size_t> out_shape = input.shape();
    if (axis < 0) axis += out_shape.size();
    if (!keepdims) {
        out_shape.erase(out_shape.begin() + axis);
    } else {
        out_shape[axis] = 1;
    }
    CUDATensor out(out_shape);
    cuda_sum(input.data(), axis, keepdims, out.data(), input.shape().data(), input.shape().size());
    return out;
}

CUDATensor cuda_max_wrapper(const CUDATensor& input, int axis, bool keepdims) {
    std::vector<size_t> out_shape = input.shape();
    if (axis < 0) axis += out_shape.size();
    if (!keepdims) {
        out_shape.erase(out_shape.begin() + axis);
    } else {
        out_shape[axis] = 1;
    }
    CUDATensor out(out_shape);
    cuda_max(input.data(), axis, keepdims, out.data(), input.shape().data(), input.shape().size());
    return out;
}

CUDATensor cuda_argmax_wrapper(const CUDATensor& input, int axis) {
    std::vector<size_t> out_shape = input.shape();
    if (axis < 0) axis += out_shape.size();
    out_shape.erase(out_shape.begin() + axis);
    CUDATensor out(out_shape);
    
    // Allocate temporary int array for GPU
    int* d_output;
    cudaMalloc(&d_output, get_total_size(out_shape) * sizeof(int));
    
    cuda_argmax(input.data(), axis, d_output, input.shape().data(), input.shape().size());
    
    // Convert int indices to float
    std::vector<int> cpu_output(get_total_size(out_shape));
    cudaMemcpy(cpu_output.data(), d_output, cpu_output.size() * sizeof(int), cudaMemcpyDeviceToHost);
    
    float* out_ptr = out.data();
    for (size_t i = 0; i < cpu_output.size(); i++) {
        out_ptr[i] = static_cast<float>(cpu_output[i]);
    }
    
    cudaFree(d_output);
    return out;
}

CUDATensor cuda_broadcast_wrapper(const CUDATensor& input, const std::vector<size_t>& target_shape) {
    CUDATensor out(target_shape);
    cuda_broadcast(input.data(), input.shape().data(), out.data(), target_shape.data(), target_shape.size());
    return out;
}

CUDATensor cuda_pow_wrapper(const CUDATensor& input, float power) {
    CUDATensor out(input.shape());
    cuda_pow(input.data(), power, out.data(), get_total_size(input.shape()));
    return out;
}

CUDATensor cuda_transpose_wrapper(const CUDATensor& input) {
    if (input.shape().size() != 2) {
        throw std::runtime_error("Input must be a 2D array");
    }
    
    std::vector<size_t> out_shape = {input.shape()[1], input.shape()[0]};
    CUDATensor out(out_shape);
    cuda_transpose(input.data(), out.data(), input.shape()[0], input.shape()[1]);
    return out;
}

PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "CUDA operations for Orange Autograd"; // module docstring
    
    // Bind CUDATensor class
    py::class_<CUDATensor>(m, "CUDATensor")
        .def(py::init<const std::vector<size_t>&>())
        .def("to_cpu", &CUDATensor::to_cpu)
        .def_property_readonly("shape", &CUDATensor::shape_tuple);
    
    m.def("create_cuda_tensor", &create_cuda_tensor, "Create a CUDA tensor from numpy array",
          py::arg("input"));
    
    m.def("add", &cuda_add_wrapper, "Add two CUDA tensors",
          py::arg("a"), py::arg("b"));
    
    m.def("mul", &cuda_mul_wrapper, "Multiply two CUDA tensors",
          py::arg("a"), py::arg("b"));
    
    m.def("matmul", &cuda_matmul_wrapper, "Matrix multiplication of CUDA tensors",
          py::arg("a"), py::arg("b"));
    
    m.def("relu", &cuda_relu_wrapper, "ReLU activation on CUDA tensor",
          py::arg("input"));
    
    m.def("softmax", &cuda_softmax_wrapper, "Softmax activation on CUDA tensor",
          py::arg("input"));
    
    m.def("exp", &cuda_exp_wrapper, "Exponential on CUDA tensor",
          py::arg("input"));
    
    m.def("log", &cuda_log_wrapper, "Natural logarithm on CUDA tensor",
          py::arg("input"));
    
    m.def("div", &cuda_div_wrapper, "Divide CUDA tensor by scalar",
          py::arg("input"), py::arg("scalar"));
    
    m.def("sub", &cuda_sub_wrapper, "Subtract two CUDA tensors",
          py::arg("a"), py::arg("b"));
    
    m.def("onehot", &cuda_onehot_wrapper, "Create one-hot encoding on CUDA",
          py::arg("indices"), py::arg("num_classes"));
    
    m.def("sum", &cuda_sum_wrapper, "Sum CUDA tensor along axis",
          py::arg("input"), py::arg("axis"), py::arg("keepdims"));
    
    m.def("max", &cuda_max_wrapper, "Maximum of CUDA tensor along axis",
          py::arg("input"), py::arg("axis"), py::arg("keepdims"));
    
    m.def("argmax", &cuda_argmax_wrapper, "Argmax of CUDA tensor along axis",
          py::arg("input"), py::arg("axis"));
    
    m.def("broadcast", &cuda_broadcast_wrapper, "Broadcast CUDA tensor to target shape",
          py::arg("input"), py::arg("target_shape"));
    
    m.def("pow", &cuda_pow_wrapper, "Power operation on CUDA tensor",
          py::arg("input"), py::arg("power"));
    
    m.def("transpose", &cuda_transpose_wrapper, "Transpose a CUDA tensor",
          py::arg("input"));
} 