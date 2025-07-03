#!/bin/bash
# compile_hyperbolic_kernel.sh

# Compilation script for the fused hyperbolic convolution kernel

set -e

OPTIMAL_NVCC_FLAGS="-shared -Xcompiler -fPIC -O3 -use_fast_math"
DEBUGGING_NVCC_FLAGS="-g -G -shared -Xcompiler -fPIC -O0 -use_fast_math"
CUDA_ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
LIBRARIES="-lcublas"

# nvcc $OPTIMIAL_NVCC_FLAGS $CUDA_ARCH -o hyperbolic_conv2d_kernel.so hyperbolic_conv2d_kernel.cu $LIBRARIES
nvcc $DEBUGGING_NVCC_FLAGS $CUDA_ARCH -o hyperbolic_conv2d_kernel.so hyperbolic_conv2d_kernel.cu $LIBRARIES