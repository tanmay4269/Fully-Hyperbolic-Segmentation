import torch
import torch.nn as nn
import numpy as np
import ctypes
import os
from typing import Optional

class FusedHyperbolicConv2d(nn.Module):
    """
    Fused CUDA implementation of Hyperbolic Conv2D using custom kernel.
    Eliminates CPU-GPU round trips and intermediate memory allocations.
    """
    
    def __init__(
        self,
        manifold_k: float,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        device: str = 'cuda'
    ):
        super(FusedHyperbolicConv2d, self).__init__()
        
        self.manifold_k = manifold_k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        
        # Calculate linearized feature dimensions
        self.linearized_features = (in_channels - 1) * kernel_size * kernel_size + 1
        
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.randn(
            out_channels, self.linearized_features, 
            device=device, dtype=torch.float32
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device=device, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
            
        # Load CUDA library
        self._load_cuda_lib()
        
        # Initialize weights
        self.reset_parameters()
        
    def _load_cuda_lib(self):
        """Load the compiled CUDA library."""
        # You'll need to compile the CUDA code first
        lib_path = os.path.join(os.path.dirname(__file__), 'hyperbolic_conv2d_kernel.so')
        
        if not os.path.exists(lib_path):
            raise RuntimeError(f"CUDA library not found at {lib_path}. Please compile first.")
            
        self.cuda_lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.cuda_lib.launch_hyperbolic_conv2d_kernel.argtypes = [
            ctypes.c_void_p,  # input
            ctypes.c_void_p,  # weight
            ctypes.c_void_p,  # bias
            ctypes.c_void_p,  # output
            ctypes.c_int,     # batch_size
            ctypes.c_int,     # in_height
            ctypes.c_int,     # in_width
            ctypes.c_int,     # in_channels
            ctypes.c_int,     # out_channels
            ctypes.c_int,     # kernel_size
            ctypes.c_int,     # stride
            ctypes.c_int,     # padding
            ctypes.c_float,   # manifold_k
            ctypes.c_void_p   # stream
        ]
        
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization adapted for hyperbolic space."""
        # Standard deviation for hyperbolic initialization
        std = np.sqrt(2.0 / (self.in_channels - 1) / (self.kernel_size ** 2))
        
        with torch.no_grad():
            self.weight.uniform_(-std, std)
            if self.bias is not None:
                self.bias.uniform_(-std, std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using fused CUDA kernel.
        
        Args:
            x: Input tensor of shape [batch, height, width, channels]
            
        Returns:
            Output tensor of shape [batch, out_height, out_width, out_channels]
        """
        if not x.is_cuda:
            raise ValueError("Input tensor must be on CUDA device")
            
        if x.dtype != torch.float32:
            x = x.float()
            
        batch_size, in_height, in_width, in_channels = x.shape
        
        if in_channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {in_channels}")
            
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Allocate output tensor
        output = torch.empty(
            batch_size, out_height, out_width, self.out_channels,
            device=x.device, dtype=torch.float32
        )
        
        # Get tensor data pointers
        input_ptr = x.data_ptr()
        weight_ptr = self.weight.data_ptr()
        bias_ptr = self.bias.data_ptr() if self.bias is not None else None
        output_ptr = output.data_ptr()
        
        # Get current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        
        # Launch kernel
        self.cuda_lib.launch_hyperbolic_conv2d_kernel(
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            batch_size,
            in_height,
            in_width,
            in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.manifold_k,
            stream
        )
        
        # Synchronize to catch any errors
        torch.cuda.synchronize()
        
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (f'in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, manifold_k={self.manifold_k}')

