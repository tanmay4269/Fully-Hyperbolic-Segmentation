// hyperbolic_conv2d_kernel.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Forward declarations
extern "C" {
    void launch_hyperbolic_conv2d_kernel(
        float* input,           // [batch, height, width, in_channels]
        float* weight,          // [out_channels, linearized_features]
        float* bias,            // [out_channels] or nullptr
        float* output,          // [batch, out_height, out_width, out_channels]
        int batch_size,
        int in_height,
        int in_width,
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        float manifold_k,
        cudaStream_t stream
    );
}

// Device function for hyperbolic constraint enforcement
__device__ __forceinline__ float enforce_hyperbolic_constraint(float* patch_time, int kernel_area, float k) {
    float sum_squares = 0.0f;
    #pragma unroll
    for (int i = 0; i < kernel_area; i++) {
        sum_squares += patch_time[i] * patch_time[i];
    }
    return sqrtf(fmaxf(sum_squares - (kernel_area - 1) * k, k));
}

// Main fused hyperbolic convolution kernel
__global__ void hyperbolic_conv2d_kernel(
    const float* __restrict__ input,     // [batch, height, width, in_channels]
    const float* __restrict__ weight,    // [out_channels, linearized_features]
    const float* __restrict__ bias,      // [out_channels]
    float* __restrict__ output,          // [batch, out_height, out_width, out_channels]
    int batch_size,
    int in_height,
    int in_width,
    int in_channels,
    int out_height,
    int out_width,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float manifold_k
) {
    // Thread indexing
    int batch_idx = blockIdx.z;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_y >= out_height || out_x >= out_width) {
        return;
    }
    
    // Calculate input patch coordinates
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    int kernel_area = kernel_size * kernel_size;
    int linearized_features = (in_channels - 1) * kernel_area + 1;
    
    // Shared memory for patch data and intermediate results
    extern __shared__ float shared_mem[];
    float* patch_time = shared_mem;
    float* patch_space = patch_time + kernel_area;
    float* linearized_patch = patch_space + (in_channels - 1) * kernel_area;
    
    // Extract patch and separate time/space components
    for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
            int patch_idx = ky * kernel_size + kx;
            int in_y = in_y_start + ky;
            int in_x = in_x_start + kx;
            
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int input_idx = ((batch_idx * in_height + in_y) * in_width + in_x) * in_channels;
                
                // Time component (first channel)
                patch_time[patch_idx] = fmaxf(input[input_idx], sqrtf(manifold_k));
                
                // Space components (remaining channels)
                for (int c = 1; c < in_channels; c++) {
                    patch_space[patch_idx * (in_channels - 1) + (c - 1)] = input[input_idx + c];
                }
            } else {
                // Padding with hyperbolic origin
                patch_time[patch_idx] = sqrtf(manifold_k);
                for (int c = 0; c < in_channels - 1; c++) {
                    patch_space[patch_idx * (in_channels - 1) + c] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Perform Lorentz direct concatenation
    float rescaled_time = enforce_hyperbolic_constraint(patch_time, kernel_area, manifold_k);
    
    // Build linearized patch: [rescaled_time, flattened_space_components]
    linearized_patch[0] = rescaled_time;
    for (int i = 0; i < (in_channels - 1) * kernel_area; i++) {
        linearized_patch[i + 1] = patch_space[i];
    }
    
    __syncthreads();
    
    // Compute output for all output channels
    for (int oc = 0; oc < out_channels; oc++) {
        float result = 0.0f;
        
        // Matrix multiplication: weight[oc] * linearized_patch
        #pragma unroll 8
        for (int f = 0; f < linearized_features; f++) {
            result += weight[oc * linearized_features + f] * linearized_patch[f];
        }
        
        // Add bias if present
        if (bias != nullptr) {
            result += bias[oc];
        }
        
        // Ensure output satisfies hyperbolic constraint
        if (oc == 0) {
            result = fmaxf(result, sqrtf(manifold_k));
        }
        
        // Write output
        int output_idx = ((batch_idx * out_height + out_y) * out_width + out_x) * out_channels + oc;
        output[output_idx] = result;
    }
}

// Host function to launch kernel
void launch_hyperbolic_conv2d_kernel(
    float* input,
    float* weight,
    float* bias,
    float* output,
    int batch_size,
    int in_height,
    int in_width,
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding,
    float manifold_k,
    cudaStream_t stream
) {
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Thread block configuration
    dim3 block_size(16, 16);  // 256 threads per block
    dim3 grid_size(
        (out_width + block_size.x - 1) / block_size.x,
        (out_height + block_size.y - 1) / block_size.y,
        batch_size
    );
    
    // Calculate shared memory requirements
    int kernel_area = kernel_size * kernel_size;
    int linearized_features = (in_channels - 1) * kernel_area + 1;
    size_t shared_mem_size = (
        kernel_area +                           // patch_time
        (in_channels - 1) * kernel_area +       // patch_space
        linearized_features                     // linearized_patch
    ) * sizeof(float);
    
    // Launch kernel
    hyperbolic_conv2d_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
        input, weight, bias, output,
        batch_size, in_height, in_width, in_channels,
        out_height, out_width, out_channels,
        kernel_size, stride, padding, manifold_k
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

// Utility function for memory allocation and management
extern "C" {
    void* cuda_malloc(size_t size) {
        void* ptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }
    
    void cuda_free(void* ptr) {
        cudaFree(ptr);
    }
    
    void cuda_memcpy_h2d(void* dst, const void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    void cuda_memcpy_d2h(void* dst, const void* src, size_t size) {
        cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
}