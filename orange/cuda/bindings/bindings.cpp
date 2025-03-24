#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "../kernels/tensor_ops.cu"

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
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }
        check_cuda_error(cudaMalloc(&data_, size * sizeof(float)));
    }

    CUDATensor(const std::vector<size_t>& shape, const float* host_data) : shape_(shape) {
        size_t size = 1;
        for (size_t dim : shape) {
            size *= dim;
        }
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

    // Copy data to host
    py::array_t<float> to_cpu() const {
        size_t size = 1;
        for (size_t dim : shape_) {
            size *= dim;
        }
        
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

// Function to create CUDA tensor from numpy array
CUDATensor create_cuda_tensor(py::array_t<float> input) {
    py::buffer_info input_buf = input.request();
    std::vector<size_t> shape(input_buf.shape.begin(), input_buf.shape.end());
    return CUDATensor(shape, (float*)input_buf.ptr);
}

// CUDA tensor operations
CUDATensor cuda_add_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shapes must match");
    }
    
    CUDATensor out(a.shape());
    cuda_add(a.data(), b.data(), out.data(), a.shape());
    return out;
}

CUDATensor cuda_mul_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shapes must match");
    }
    
    CUDATensor out(a.shape());
    cuda_mul(a.data(), b.data(), out.data(), a.shape());
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
    cuda_relu(input.data(), out.data(), input.shape());
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

// Additional CUDA tensor operations
CUDATensor cuda_exp_wrapper(const CUDATensor& input) {
    CUDATensor out(input.shape());
    cuda_exp(input.data(), out.data(), input.shape());
    return out;
}

CUDATensor cuda_log_wrapper(const CUDATensor& input) {
    CUDATensor out(input.shape());
    cuda_log(input.data(), out.data(), input.shape());
    return out;
}

CUDATensor cuda_div_wrapper(const CUDATensor& a, float b) {
    CUDATensor out(a.shape());
    cuda_div(a.data(), b, out.data(), a.shape());
    return out;
}

CUDATensor cuda_sub_wrapper(const CUDATensor& a, const CUDATensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("Shapes must match");
    }
    CUDATensor out(a.shape());
    cuda_sub(a.data(), b.data(), out.data(), a.shape());
    return out;
}

CUDATensor cuda_onehot_wrapper(const CUDATensor& indices, size_t num_classes) {
    std::vector<size_t> shape = indices.shape();
    shape.push_back(num_classes);
    CUDATensor out(shape);
    cuda_onehot(indices.data(), num_classes, out.data(), indices.shape());
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
    cuda_sum(input.data(), axis, keepdims, out.data(), input.shape());
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
    cuda_max(input.data(), axis, keepdims, out.data(), input.shape());
    return out;
}

CUDATensor cuda_argmax_wrapper(const CUDATensor& input, int axis) {
    std::vector<size_t> out_shape = input.shape();
    if (axis < 0) axis += out_shape.size();
    out_shape.erase(out_shape.begin() + axis);
    CUDATensor out(out_shape);
    cuda_argmax(input.data(), axis, out.data(), input.shape());
    return out;
}

CUDATensor cuda_broadcast_wrapper(const CUDATensor& input, const std::vector<size_t>& target_shape) {
    CUDATensor out(target_shape);
    cuda_broadcast(input.data(), target_shape, out.data(), input.shape());
    return out;
}

PYBIND11_MODULE(cuda_ops, m) {
    m.doc() = "CUDA operations for Orange Autograd"; // module docstring
    
    // Bind CUDATensor class
    py::class_<CUDATensor>(m, "CUDATensor")
        .def(py::init<const std::vector<size_t>&>())
        .def("to_cpu", &CUDATensor::to_cpu)
        .def_property_readonly("shape", &CUDATensor::shape);
    
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
} 