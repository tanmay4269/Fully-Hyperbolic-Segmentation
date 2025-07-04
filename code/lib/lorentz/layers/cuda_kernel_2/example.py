import torch
from lib.lorentz.manifold import CustomLorentz
from wrapper import FusedLorentzConv2d

# Initialize
manifold = CustomLorentz(k=1.0)
conv_layer = FusedLorentzConv2d(
    manifold=manifold,
    in_channels=3,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True
).cuda()

# Input
x = torch.randn(4, 224, 224, 3).cuda()

# Forward pass
output = conv_layer(x)
print(f"Output shape: {output.shape}")  # Should be [4, 224, 224, 65]