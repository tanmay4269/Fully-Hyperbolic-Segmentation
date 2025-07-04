// fused_lorentz_conv2d.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile dimensions for thread block configuration.
#define TILE_DIM 16

__global__ void fused_lorentz_conv2d_kernel_tiled(
    const float* __restrict__ input,     // [B, H, W, C_in]
    const float* __restrict__ weight,    // [C_out, in_features]
    const float* __restrict__ bias,      // [C_out] or nullptr
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
    // --- Dynamic Shared Memory Declaration ---
    // Declare a single, unsized extern array. Its size is provided at launch.
    extern __shared__ float s_mem[];

    // Manually partition the memory: weights first, then the input tile.
    float* s_weights = s_mem;
    float* s_input = &s_mem[weight_features]; // Input tile starts after weights

    // --- Thread Indexing ---
    const int out_x_base = blockIdx.x * TILE_DIM;
    const int out_y_base = blockIdx.y * TILE_DIM;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_idx_flat = ty * TILE_DIM + tx;

    const int batch_idx = blockIdx.z / out_channels;
    const int out_ch = blockIdx.z % out_channels;

    const int out_x = out_x_base + tx;
    const int out_y = out_y_base + ty;

    // --- Cooperative Loading of Weights ---
    const int weight_row_start = out_ch * weight_features;
    for (int i = thread_idx_flat; i < weight_features; i += TILE_DIM * TILE_DIM) {
        s_weights[i] = weight[weight_row_start + i];
    }

    // --- Cooperative Loading of Input Tile (Global -> Shared) ---
    const int tile_h = TILE_DIM * stride_h + kernel_h - 1;
    const int tile_w = TILE_DIM * stride_w + kernel_w - 1;
    const int in_x_origin = out_x_base * stride_w - padding_w;
    const int in_y_origin = out_y_base * stride_h - padding_h;
    const int tile_size_elements = tile_h * tile_w * in_channels;

    __syncthreads(); // Wait for weights to be loaded

    for (int i = thread_idx_flat; i < tile_size_elements; i += TILE_DIM * TILE_DIM) {
        int c = i % in_channels;
        int x = (i / in_channels) % tile_w;
        int y = i / (in_channels * tile_w);

        int in_glob_y = in_y_origin + y;
        int in_glob_x = in_x_origin + x;

        if (in_glob_y >= 0 && in_glob_y < in_height && in_glob_x >= 0 && in_glob_x < in_width) {
            int global_idx = ((batch_idx * in_height + in_glob_y) * in_width + in_glob_x) * in_channels + c;
            s_input[i] = input[global_idx];
        } else {
            s_input[i] = (c == 0) ? sqrtf(k_value) : 0.0f;
        }
    }
    
    __syncthreads(); // Wait for input tile to be loaded

    // --- Computation (from Shared Memory) ---
    if (out_y < out_height && out_x < out_width) {
        float time_sum_sq = 0.0f;
        float space_result = 0.0f;
        int space_weight_idx = 1;

        const int in_s_y_start = ty * stride_h;
        const int in_s_x_start = tx * stride_w;

        for (int ky = 0; ky < kernel_h; ky++) {
            for (int kx = 0; kx < kernel_w; kx++) {
                const int in_s_y = in_s_y_start + ky * dilation_h;
                const int in_s_x = in_s_x_start + kx * dilation_w;
                const int s_idx_base = (in_s_y * tile_w + in_s_x) * in_channels;

                float time_val = s_input[s_idx_base];
                time_sum_sq += time_val * time_val;

                for (int ch = 1; ch < in_channels; ch++) {
                    float space_val = s_input[s_idx_base + ch];
                    space_result += space_val * s_weights[space_weight_idx];
                    space_weight_idx++;
                }
            }
        }

        float aggregated_time = sqrtf(fmaxf(0.0f, time_sum_sq - (float)(kernel_h * kernel_w - 1) * k_value));
        float time_result = aggregated_time * s_weights[0];
        float final_result = time_result + space_result;

        if (bias != nullptr) {
            final_result += bias[out_ch];
        }

        int output_idx = ((batch_idx * out_height + out_y) * out_width + out_x) * out_channels + out_ch;
        output[output_idx] = final_result;
    }
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
    
    const int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    const int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::empty({batch_size, out_height, out_width, out_channels}, 
                              torch::TensorOptions().dtype(input.dtype()).device(input.device()));
    
    // --- Updated Kernel Launch Configuration ---
    dim3 block_dim(TILE_DIM, TILE_DIM, 1);
    dim3 grid_dim(
        (out_width + TILE_DIM - 1) / TILE_DIM,
        (out_height + TILE_DIM - 1) / TILE_DIM,
        batch_size * out_channels
    );

    // --- Calculate Dynamic Shared Memory Size ---
    const int tile_h = TILE_DIM * stride_h + kernel_h - 1;
    const int tile_w = TILE_DIM * stride_w + kernel_w - 1;
    size_t weights_size_bytes = weight_features * sizeof(float);
    size_t input_tile_size_bytes = tile_h * tile_w * in_channels * sizeof(float);
    size_t total_shared_mem = weights_size_bytes + input_tile_size_bytes;

    // Optional: Check against device limits
    // int device;
    // cudaGetDevice(&device);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, device);
    // TORCH_CHECK(total_shared_mem < prop.sharedMemPerBlock, "Requested shared memory exceeds device limits");

    fused_lorentz_conv2d_kernel_tiled<<<grid_dim, block_dim, total_shared_mem>>>(
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
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_lorentz_conv2d_cuda", &fused_lorentz_conv2d_cuda, "Fused Lorentz Conv2d CUDA (Tiled, Dynamic Shared)");
}