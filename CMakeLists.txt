cmake_minimum_required(VERSION 3.18)
project(orange_cuda)

# Find CUDA
find_package(CUDA REQUIRED)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Add CUDA kernels
cuda_add_library(orange_cuda_kernels
    orange/cuda/kernels/tensor_ops.cu
)

# Add pybind11 bindings
pybind11_add_module(orange_cuda_bindings
    orange/cuda/bindings/bindings.cpp
)

# Link libraries
target_link_libraries(orange_cuda_bindings PRIVATE
    orange_cuda_kernels
)

# Set include directories
target_include_directories(orange_cuda_bindings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Set output directories
set_target_properties(orange_cuda_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/orange"
    PREFIX ""
    SUFFIX ".so"
) 