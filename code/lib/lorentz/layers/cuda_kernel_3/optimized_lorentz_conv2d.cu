// optimized_lorentz_conv2d.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Kernel 1: Time aggregation only (no nested loops per thread)
__global__ void time_aggregation_kernel(
    const float* __restrict__ input,        // [B, H, W, C]
    float* __restrict__ time_patches,       // [B, out_H, out_W, kernel_h*kernel_w]
    const int batch_size,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int padding_h,
    const int padding_w,
    const int dilation_h,
    const int dilation_w,
    const float k_value
) {
    // Each thread handles one element of one patch
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_height * out_width * kernel_h * kernel_w;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    const int batch_idx = idx / (out_height * out_width * kernel_h * kernel_w);
    const int remainder1 = idx % (out_height * out_width * kernel_h * kernel_w);
    const int out_y = remainder1 / (out_width * kernel_h * kernel_w);
    const int remainder2 = remainder1 % (out_width * kernel_h * kernel_w);
    const int out_x = remainder2 / (kernel_h * kernel_w);
    const int remainder3 = remainder2 % (kernel_h * kernel_w);
    const int ky = remainder3 / kernel_w;
    const int kx = remainder3 % kernel_w;
    
    // Calculate input position
    const int in_y = out_y * stride_h - padding_h + ky * dilation_h;
    const int in_x = out_x * stride_w - padding_w + kx * dilation_w;
    
    float time_val;
    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
        // Read time component (channel 0)
        const int input_idx = ((batch_idx * in_height + in_y) * in_width + in_x) * 1; // Only time channel
        time_val = input[input_idx];
        time_val = fmaxf(time_val, sqrtf(k_value)); // Clamp
    } else {
        time_val = sqrtf(k_value); // Zero padding
    }
    
    // Store time value
    const int patch_idx = ((batch_idx * out_height + out_y) * out_width + out_x) * (kernel_h * kernel_w) + (ky * kernel_w + kx);
    time_patches[patch_idx] = time_val;
}

// Kernel 2: Time reduction (parallel reduction per patch)
__global__ void time_reduction_kernel(
    const float* __restrict__ time_patches,  // [B, out_H, out_W, kernel_h*kernel_w]
    const float* __restrict__ time_weight,   // [out_channels, 1]
    float* __restrict__ time_output,         // [B, out_H, out_W, out_channels]
    const int batch_size,
    const int out_height,
    const int out_width,
    const int out_channels,
    const int kernel_size,
    const float k_value
) {
    // Each thread handles one output channel for one spatial position
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * out_height * out_width * out_channels;
    
    if (idx >= total_elements) return;
    
    // Decompose index
    const int batch_idx = idx / (out_height * out_width * out_channels);
    const int remainder1 = idx % (out_height * out_width * out_channels);
    const int out_y = remainder1 / (out_width * out_channels);
    const int remainder2 = remainder1 % (out_width * out_channels);
    const int out_x = remainder2 / out_channels;
    const int out_ch = remainder2 % out_channels;
    
    // Load time patch
    const int patch_base = ((batch_idx * out_height + out_y) * out_width + out_x) * kernel_size;
    
    // Aggregate time values (small loop - only kernel_size elements)
    float time_sum_sq = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        float time_val = time_patches[patch_base + i];
        time_sum_sq += time_val * time_val;
    }
    
    // Lorentz aggregation
    float aggregated_time = sqrtf(time_sum_sq - (kernel_size - 1) * k_value);
    
    // Apply time weight
    float time_result = aggregated_time * time_weight[out_ch];
    
    // Store result
    time_output[idx] = time_result;
}

// Host functions
torch::Tensor optimized_lorentz_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int> kernel_size,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    float k_value
) {
    const int batch_size = input.size(0);
    const int in_height = input.size(1);
    const int in_width = input.size(2);
    const int in_channels = input.size(3);
    const int out_channels = weight.size(0);
    
    const int kernel_h = kernel_size[0];
    const int kernel_w = kernel_size[1];
    const int kernel_total = kernel_h * kernel_w;
    
    // Calculate output dimensions
    const int out_height = (in_height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) / stride[0] + 1;
    const int out_width = (in_width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) / stride[1] + 1;
    
    // Split input into time and space components
    torch::Tensor time_input = input.slice(3, 0, 1);  // [B, H, W, 1]
    torch::Tensor space_input = input.slice(3, 1, in_channels);  // [B, H, W, C-1]
    
    // Space convolution using cuDNN (highly optimized)
    torch::Tensor space_input_chw = space_input.permute({0, 3, 1, 2});  // BCHW
    torch::Tensor space_weight = weight.slice(1, 1, weight.size(1)).view({out_channels, in_channels-1, kernel_h, kernel_w});
    torch::Tensor space_output = torch::conv2d(space_input_chw, space_weight, torch::nullopt, stride, padding, dilation);
    space_output = space_output.permute({0, 2, 3, 1});  // Back to BHWC
    
    // Time processing using custom kernels
    auto time_patches = torch::zeros({batch_size, out_height, out_width, kernel_total}, input.options());
    auto time_weight = weight.slice(1, 0, 1);  // [out_channels, 1]
    auto time_output = torch::zeros({batch_size, out_height, out_width, out_channels}, input.options());
    
    // Launch time aggregation kernel
    int total_patch_elements = batch_size * out_height * out_width * kernel_total;
    int block_size = BLOCK_SIZE;
    int grid_size = (total_patch_elements + block_size - 1) / block_size;
    
    time_aggregation_kernel<<<grid_size, block_size>>>(
        time_input.data_ptr<float>(),
        time_patches.data_ptr<float>(),
        batch_size, in_height, in_width, out_height, out_width,
        kernel_h, kernel_w, stride[0], stride[1], padding[0], padding[1],
        dilation[0], dilation[1], k_value
    );
    
    // Launch time reduction kernel
    int total_output_elements = batch_size * out_height * out_width * out_channels;
    grid_size = (total_output_elements + block_size - 1) / block_size;
    
    time_reduction_kernel<<<grid_size, block_size>>>(
        time_patches.data_ptr<float>(),
        time_weight.data_ptr<float>(),
        time_output.data_ptr<float>(),
        batch_size, out_height, out_width, out_channels,
        kernel_total, k_value
    );
    
    // Combine space and time results
    torch::Tensor combined_output = space_output + time_output;
    
    // Add bias if present
    if (bias.defined()) {
        combined_output = combined_output + bias;
    }
    
    return combined_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("optimized_lorentz_conv2d_cuda", &optimized_lorentz_conv2d_cuda, "Optimized Lorentz Conv2d CUDA");
}