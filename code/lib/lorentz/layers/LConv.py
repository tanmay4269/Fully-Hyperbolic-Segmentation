import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzFullyConnected

class LorentzConv1d(nn.Module):
    """ Implements a fully hyperbolic 1D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, bias: Same as nn.Conv1d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv1d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        lin_features = (self.in_channels - 1) * self.kernel_size + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x len x C """
        bsz = x.shape[0]

        # origin padding
        x = F.pad(x, (0, 0, self.padding, self.padding))
        x[..., 0].clamp_(min=self.manifold.k.sqrt()) 

        patches = x.unfold(1, self.kernel_size, self.stride)
        # Lorentz direct concatenation of features within patches
        patches_time = patches.narrow(2, 0, 1)
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=(-2,-1), keepdim=True) - ((self.kernel_size - 1) * self.manifold.k))
        patches_time_rescaled = patches_time_rescaled.view(bsz, patches.shape[1], -1)

        patches_space = patches.narrow(2, 1, patches.shape[2]-1).reshape(bsz, patches.shape[1], -1)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)

        return out


class LorentzConv2d(nn.Module):
    """ Implements a fully hyperbolic 2D convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias: Same as nn.Conv2d (dilation not tested)
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self,
            manifold: CustomLorentz,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            LFC_normalize=False
    ):
        super(LorentzConv2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation

        self.kernel_len = self.kernel_size[0] * self.kernel_size[1]

        lin_features = ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]) + 1

        self.linearized_kernel = LorentzFullyConnected(
            manifold,
            lin_features, 
            self.out_channels, 
            bias=bias,
            normalize=LFC_normalize
        )
        self.unfold = torch.nn.Unfold(kernel_size=(self.kernel_size[0], self.kernel_size[1]), dilation=dilation, padding=padding, stride=stride)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels-1) * self.kernel_size[0] * self.kernel_size[1]))
        self.linearized_kernel.weight.weight.data.uniform_(-stdv, stdv)
        if self.bias:
            self.linearized_kernel.weight.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bsz = x.shape[0]
        h, w = x.shape[1:3]

        h_out = math.floor(
            (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = math.floor(
            (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        x = x.permute(0, 3, 1, 2)

        patches = self.unfold(x)  # batch_size, channels * elements/window, windows
        patches = patches.permute(0, 2, 1)

        # Now we have flattened patches with multiple time elements -> fix the concatenation to perform Lorentz direct concatenation by Qu et al. (2022)
        patches_time = torch.clamp(patches.narrow(-1, 0, self.kernel_len), min=self.manifold.k.sqrt())  # Fix zero (origin) padding
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((self.kernel_len - 1) * self.manifold.k))

        patches_space = patches.narrow(-1, self.kernel_len, patches.shape[-1] - self.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], self.in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape) # No need, but seems to improve runtime??

        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)

        out = self.linearized_kernel(patches_pre_kernel)
        out = out.view(bsz, h_out, w_out, self.out_channels)

        return out

class LorentzConvTranspose2d(nn.Module):
    """ Implements a fully hyperbolic 2D transposed convolutional layer using the Lorentz model.

    Args:
        manifold: Instance of Lorentz manifold
        in_channels, out_channels, kernel_size, stride, padding, output_padding, bias: Same as nn.ConvTranspose2d
        LFC_normalize: If Chen et al.'s internal normalization should be used in LFC 
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            output_padding=0, 
            bias=True,
            LFC_normalize=False
        ):
        super(LorentzConvTranspose2d, self).__init__()

        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding

        padding_implicit = [0,0]
        padding_implicit[0] = kernel_size - self.padding[0] - 1 # Ensure padding > kernel_size
        padding_implicit[1] = kernel_size - self.padding[1] - 1 # Ensure padding > kernel_size

        self.pad_weight = nn.Parameter(F.pad(torch.ones((self.in_channels,1,1,1)),(1,1,1,1)), requires_grad=False)

        self.conv = LorentzConv2d(
            manifold=manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding_implicit, 
            bias=bias, 
            LFC_normalize=LFC_normalize
        )

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        if self.stride[0] > 1 or self.stride[1] > 1:
            # Insert hyperbolic origin vectors between features
            x = x.permute(0,3,1,2)
            # -> Insert zero vectors
            x = F.conv_transpose2d(x, self.pad_weight,stride=self.stride,padding=1, groups=self.in_channels)
            x = x.permute(0,2,3,1)
            x[..., 0].clamp_(min=self.manifold.k.sqrt())

        x = self.conv(x)

        if self.output_padding[0] > 0 or self.output_padding[1] > 0:
            x = F.pad(x, pad=(0, self.output_padding[1], 0, self.output_padding[0])) # Pad one side of each dimension (bottom+right) (see PyTorch documentation)
            x[..., 0].clamp_(min=self.manifold.k.sqrt()) # Fix origin padding

        return x

def profile_lorentz_conv2d(manifold, in_channels, out_channels, kernel_size, input_size, 
                          stride=1, padding=0, dilation=1, bias=True, LFC_normalize=False, 
                          iterations=100, warmup=10):
    """
    Profiles the LorentzConv2d layer with detailed CUDA timing.
    
    Args:
        manifold: Instance of Lorentz manifold
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        input_size: Tuple of (height, width) for input tensor
        stride: Stride of convolution
        padding: Padding of convolution
        dilation: Dilation of convolution
        bias: Whether to use bias
        LFC_normalize: Whether to use normalization in LFC
        iterations: Number of iterations for profiling
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary containing profiling results for each operation
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Cannot perform CUDA profiling.")
    
    # Create model and move to GPU
    conv = LorentzConv2d(
        manifold=manifold,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        LFC_normalize=LFC_normalize
    ).cuda()
    
    # Create input tensor
    batch_size = 16
    h, w = input_size
    
    # Create a valid Lorentz point for the first coordinate
    x = torch.randn(batch_size, h, w, in_channels).cuda()
    x[..., 0] = torch.sqrt(manifold.k + torch.sum(x[..., 1:] ** 2, dim=-1))
    
    # Warmup
    for _ in range(warmup):
        out = conv(x)
        torch.cuda.synchronize()
    
    # Dictionary to store profiling results
    profile_results = {
        'total': 0.0,
        'unfold': 0.0,
        'patch_processing': 0.0,
        'linearized_kernel': 0.0
    }
    
    # Start profiling
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Profile total time
    start_event.record()
    for _ in range(iterations):
        out = conv(x)
    end_event.record()
    torch.cuda.synchronize()
    profile_results['total'] = start_event.elapsed_time(end_event) / iterations
    
    # Profile unfold operation
    start_event.record()
    for _ in range(iterations):
        x_permuted = x.permute(0, 3, 1, 2)
        patches = conv.unfold(x_permuted)
        patches = patches.permute(0, 2, 1)
    end_event.record()
    torch.cuda.synchronize()
    profile_results['unfold'] = start_event.elapsed_time(end_event) / iterations
    
    # Profile patch processing
    x_permuted = x.permute(0, 3, 1, 2)
    patches = conv.unfold(x_permuted)
    patches = patches.permute(0, 2, 1)
    
    start_event.record()
    for _ in range(iterations):
        patches_time = torch.clamp(patches.narrow(-1, 0, conv.kernel_len), min=manifold.k.sqrt())
        patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((conv.kernel_len - 1) * manifold.k))
        patches_space = patches.narrow(-1, conv.kernel_len, patches.shape[-1] - conv.kernel_len)
        patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)
        patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
    end_event.record()
    torch.cuda.synchronize()
    profile_results['patch_processing'] = start_event.elapsed_time(end_event) / iterations
    
    # Profile linearized kernel
    patches_time = torch.clamp(patches.narrow(-1, 0, conv.kernel_len), min=manifold.k.sqrt())
    patches_time_rescaled = torch.sqrt(torch.sum(patches_time ** 2, dim=-1, keepdim=True) - ((conv.kernel_len - 1) * manifold.k))
    patches_space = patches.narrow(-1, conv.kernel_len, patches.shape[-1] - conv.kernel_len)
    patches_space = patches_space.reshape(patches_space.shape[0], patches_space.shape[1], in_channels - 1, -1).transpose(-1, -2).reshape(patches_space.shape)
    patches_pre_kernel = torch.concat((patches_time_rescaled, patches_space), dim=-1)
    
    start_event.record()
    for _ in range(iterations):
        out = conv.linearized_kernel(patches_pre_kernel)
    end_event.record()
    torch.cuda.synchronize()
    profile_results['linearized_kernel'] = start_event.elapsed_time(end_event) / iterations
    
    # Calculate memory usage
    memory_stats = {
        'allocated': torch.cuda.memory_allocated() / (1024 ** 2),  # MB
        'reserved': torch.cuda.memory_reserved() / (1024 ** 2)     # MB
    }
    profile_results['memory'] = memory_stats
    
    # Calculate percentage of time spent in each operation
    total_time = profile_results['total']
    for key in ['unfold', 'patch_processing', 'linearized_kernel']:
        profile_results[f'{key}_percent'] = (profile_results[key] / total_time) * 100
    
    # Calculate unaccounted time
    accounted_time = profile_results['unfold'] + profile_results['patch_processing'] + profile_results['linearized_kernel']
    profile_results['unaccounted'] = total_time - accounted_time
    profile_results['unaccounted_percent'] = (profile_results['unaccounted'] / total_time) * 100
    
    return profile_results

if __name__ == "__main__":
    import sys
    import os
    import json
    from tabulate import tabulate
    
    # Add parent directory to path to import CustomLorentz
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    
    # Import the manifold
    from lib.lorentz.manifold import CustomLorentz
    
    # Create manifold
    manifold = CustomLorentz(k=1.0)
    
    # Define configurations to profile
    configs = [
        {"in_channels": 64, "out_channels": 128, "kernel_size": 3, "input_size": (32, 32)},
        {"in_channels": 128, "out_channels": 256, "kernel_size": 3, "input_size": (16, 16)},
        {"in_channels": 256, "out_channels": 512, "kernel_size": 3, "input_size": (8, 8)},
    ]
    
    # Run profiling for each configuration
    results = {}
    for i, config in enumerate(configs):
        print(f"Profiling configuration {i+1}/{len(configs)}: {config}")
        
        # Run profiling
        profile_result = profile_lorentz_conv2d(
            manifold=manifold,
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            input_size=config["input_size"],
            iterations=50,  # Reduce iterations for quicker results
            warmup=5
        )
        
        # Store results
        config_name = f"{config['in_channels']}x{config['out_channels']}_{config['input_size'][0]}x{config['input_size'][1]}"
        results[config_name] = profile_result
    
    # Print results in a table
    table_data = []
    headers = ["Configuration", "Total (ms)", "Unfold (%)", "Patch Processing (%)", "Linearized Kernel (%)", "Memory (MB)"]
    
    for config_name, result in results.items():
        table_data.append([
            config_name,
            f"{result['total']:.3f}",
            f"{result['unfold_percent']:.2f}%",
            f"{result['patch_processing_percent']:.2f}%",
            f"{result['linearized_kernel_percent']:.2f}%",
            f"{result['memory']['allocated']:.2f}"
        ])
    
    print("\nProfiling Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    with open("lorentz_conv2d_profile_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed results saved to lorentz_conv2d_profile_results.json")
