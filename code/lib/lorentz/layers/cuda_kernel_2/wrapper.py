import torch
import torch.nn as nn
import math
# import fused_lorentz_conv2d
import lib.lorentz.layers.cuda_kernel_2.fused_lorentz_conv2d as fused_lorentz_conv2d
# from lib.lorentz.layers.cuda_kernel_2.fused_lorentz_conv2d import fused_lorentz_conv2d_cuda

class FusedLorentzConv2d(nn.Module):
    def __init__(
        self,
        manifold,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        LFC_normalize=False
    ):
        super(FusedLorentzConv2d, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Normalize parameters to lists
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size]
        self.stride = stride if isinstance(stride, (list, tuple)) else [stride, stride]
        self.padding = padding if isinstance(padding, (list, tuple)) else [padding, padding]
        self.dilation = dilation if isinstance(dilation, (list, tuple)) else [dilation, dilation]
        
        # Weight matrix: [out_channels, (in_channels-1)*kernel_h*kernel_w + 1]
        # Same structure as original LorentzFullyConnected
        self.weight_features = (in_channels - 1) * self.kernel_size[0] * self.kernel_size[1] + 1
        self.weight = nn.Parameter(torch.randn(out_channels, self.weight_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Match original initialization
        stdv = math.sqrt(2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        # Input: [batch, height, width, channels] (BHWC)
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")
        
        if x.dim() != 4:
            raise RuntimeError(f"Expected 4D input (BHWC), got {x.dim()}D")
        
        # Call fused CUDA kernel - outputs space components
        space_output = fused_lorentz_conv2d.fused_lorentz_conv2d_cuda(
        # space_output = fused_lorentz_conv2d_cuda(
            x.contiguous(),
            self.weight,
            self.bias,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            float(self.manifold.k)
        )
        
        # Add time component to create proper hyperbolic vector
        output = self.manifold.add_time(space_output)
        
        return output