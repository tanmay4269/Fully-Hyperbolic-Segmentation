#!/bin/bash
# compile_hyperbolic_kernel.sh

# Compilation script for the fused hyperbolic convolution kernel

set -e

echo "Compiling Fused Hyperbolic Convolution CUDA Kernel..."

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit and add it to PATH."
    exit 1
fi

# Compilation flags
NVCC_FLAGS="-shared -Xcompiler -fPIC -O3 -use_fast_math"
CUDA_ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86"
LIBRARIES="-lcublas"

# Compile the kernel
echo "Compiling hyperbolic_conv2d_kernel.cu..."
nvcc $NVCC_FLAGS $CUDA_ARCH -o hyperbolic_conv2d_kernel.so hyperbolic_conv2d_kernel.cu $LIBRARIES

if [ $? -eq 0 ]; then
    echo "✓ Kernel compiled successfully!"
    echo "✓ Shared library: hyperbolic_conv2d_kernel.so"
else
    echo "✗ Compilation failed!"
    exit 1
fi