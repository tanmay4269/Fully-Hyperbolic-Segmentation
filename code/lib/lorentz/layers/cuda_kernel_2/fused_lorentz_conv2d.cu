// fused_lorentz_conv2d.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define BLOCK_SIZE 256
#define MAX_KERNEL_SIZE 11
#define MAX_CHANNELS 1024
#define WARP_SIZE 32

using namespace cooperative_groups;

__global__ void fused_lorentz_conv2d_kernel(
    const float* __restrict__ input,     // [B, H, W, C]
    const float* __restrict__ weight,    // [out_channels, in_features]
    const float* __restrict__ bias,      // [out_channels] or nullptr
    float* __restrict__ output,          // [B, H_out, W_out, C_out]
    const int batch_size,
    const int in_height,
    const int in_width, 
    const int in_channels,
    const int out_height,
    const int out_width,
    const int out_channels,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const float k_value,
    const int weight_features
) {
    // Each thread computes one output element
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = batch_size * out_channels * out_height * out_width;
    
    if (global_idx >= total_outputs) return;
    
    // Decompose global index
    const int batch_idx = global_idx / (out_channels * out_height * out_width);
    const int remainder1 = global_idx % (out_channels * out_height * out_width);
    const int out_ch = remainder1 / (out_height * out_width);
    const int remainder2 = remainder1 % (out_height * out_width);
    const int out_y = remainder2 / out_width;
    const int out_x = remainder2 % out_width;
    
    // Shared memory for weight caching (cooperative loading)
    __shared__ float s_weights[MAX_CHANNELS];
    const int tid = threadIdx.x;
    const int weight_row_start = out_ch * weight_features;
    
    // Cooperatively load weights for this output channel
    for (int i = tid; i < weight_features; i += blockDim.x) {
        if (i < weight_features) {
            s_weights[i] = weight[weight_row_start + i];
        }
    }
    __syncthreads();
    
    // Calculate input patch bounds
    const int in_y_start = out_y * stride_h - padding_h;
    const int in_x_start = out_x * stride_w - padding_w;
    
    // Time aggregation variables
    float time_sum_sq = 0.0f;
    int valid_pixels = 0;
    
    // Space convolution accumulator
    float space_result = 0.0f;
    int space_weight_idx = 1; // Skip time weight (index 0)
    
    // Process kernel window
    // ! Again, this is a naive implementation. 
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            const int in_y = in_y_start + ky * dilation_h;
            const int in_x = in_x_start + kx * dilation_w;
            
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                // Valid pixel - read from input
                const int input_base = ((batch_idx * in_height + in_y) * in_width + in_x) * in_channels;
                
                // Process time component (channel 0)
                float time_val = input[input_base];
                time_val = fmaxf(time_val, sqrtf(k_value)); // Clamp
                time_sum_sq += time_val * time_val;
                valid_pixels++;
                
                // Process space components (channels 1 to in_channels-1)
                for (int ch = 1; ch < in_channels; ch++) {
                    float space_val = input[input_base + ch];
                    space_result += space_val * s_weights[space_weight_idx];
                    space_weight_idx++;
                }
            } else {
                // Zero padding
                float time_val = sqrtf(k_value);
                time_sum_sq += time_val * time_val;
                valid_pixels++;
                
                // Zero space components don't contribute to space_result
                space_weight_idx += (in_channels - 1);
            }
        }
    }
    
    // Lorentz time aggregation
    float aggregated_time = sqrtf(time_sum_sq - (valid_pixels - 1) * k_value);
    
    // Apply time weight
    float time_result = aggregated_time * s_weights[0];
    
    // Combine time and space results
    float final_result = time_result + space_result;
    
    // Add bias if present
    if (bias != nullptr) {
        final_result += bias[out_ch];
    }
    
    // Write output
    output[global_idx] = final_result;
}

torch::Tensor fused_lorentz_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int> kernel_size,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    float k_value
) {
    // Input validation
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (BHWC)");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    
    const int batch_size = input.size(0);
    const int in_height = input.size(1);
    const int in_width = input.size(2);
    const int in_channels = input.size(3);
    const int out_channels = weight.size(0);
    const int weight_features = weight.size(1);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int stride_h = stride[0];
    const int stride_w = stride[1];
    const int padding_h = padding[0];
    const int padding_w = padding[1];
    const int dilation_h = dilation[0];
    const int dilation_w = dilation[1];
    
    // Calculate output dimensions
    const int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_height, out_width, out_channels}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // Launch kernel
    const int total_outputs = batch_size * out_channels * out_height * out_width;
    const int block_size = BLOCK_SIZE;
    const int grid_size = (total_outputs + block_size - 1) / block_size;
    
    fused_lorentz_conv2d_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size, in_height, in_width, in_channels,
        out_height, out_width, out_channels,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w,
        k_value, weight_features
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_lorentz_conv2d_cuda", &fused_lorentz_conv2d_cuda, "Fused Lorentz Conv2d CUDA");
}