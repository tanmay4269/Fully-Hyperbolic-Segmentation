# wrapper.py

import torch
import torch.nn as nn
import math
import lib.lorentz.layers.cuda_kernel_3.fused_lorentz_conv2d as fused_lorentz_conv2d

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
    ):
        super(FusedLorentzConv2d, self).__init__()
        
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size]
        self.stride = stride if isinstance(stride, (list, tuple)) else [stride, stride]
        self.padding = padding if isinstance(padding, (list, tuple)) else [padding, padding]
        self.dilation = dilation if isinstance(dilation, (list, tuple)) else [dilation, dilation]
        
        self.weight_features = (in_channels - 1) * self.kernel_size[0] * self.kernel_size[1] + 1
        self.weight = nn.Parameter(torch.randn(out_channels, self.weight_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = math.sqrt(2.0 / ((self.in_channels - 1) * self.kernel_size[0] * self.kernel_size[1]))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x):
        if not x.is_cuda:
            raise RuntimeError("Input must be on CUDA device")
        if x.dim() != 4 or x.shape[3] != self.in_channels:
            raise RuntimeError(f"Expected 4D input (BHWC) with {self.in_channels} channels, got {x.shape}")
        
        # The kernel computes the full output of the nn.Linear operation
        linear_out = fused_lorentz_conv2d.fused_lorentz_conv2d_cuda(
            x.contiguous(),
            self.weight,
            self.bias, # Pass self.bias directly, kernel handles nullptr
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            float(self.manifold.k)
        )
        
        # --- CORRECTED LOGIC ---
        # Replicate the logic from the original LorentzFullyConnected layer.
        # The output of the linear layer has `out_channels` features.
        # The first feature is discarded, and the remaining `out_channels - 1`
        # features form the new space vector.
        
        if self.out_channels <= 1:
             # Not a valid hyperbolic vector, but handle to avoid error
             return linear_out

        space_components = linear_out.narrow(-1, 1, self.out_channels - 1)
        
        # A new time component is calculated from the new space vector.
        # The final output will have `out_channels` dimensions.
        output = self.manifold.add_time(space_components)
        
        return output