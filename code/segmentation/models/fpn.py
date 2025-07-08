import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.profiler
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from lib.models.resnet import Lorentz_resnet18_wrapper
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR
from lib.lorentz.blocks.layer_blocks import LConv2d_Block


class FPN(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=1, pretrained=False, use_batch_norm=True):
        super().__init__()
        
        # Load backbone and get feature channels
        if backbone == 'resnet18':
            backbone_model = resnet18(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            backbone_model = resnet34(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            backbone_model = resnet50(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            backbone_model = resnet101(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone {backbone} not supported")

        self.use_batch_norm = use_batch_norm
        
        # Extract backbone layers
        self.stem = nn.Sequential(*list(backbone_model.children())[:4])
        self.layer1 = backbone_model.layer1
        self.layer2 = backbone_model.layer2
        self.layer3 = backbone_model.layer3
        self.layer4 = backbone_model.layer4
        
        # FPN components
        self.fpn_channels = 256
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral4 = nn.Conv2d(self.feature_channels[3], self.fpn_channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(self.feature_channels[2], self.fpn_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(self.feature_channels[1], self.fpn_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(self.feature_channels[0], self.fpn_channels, kernel_size=1)
        
        # Final convs after merging
        self.fpn4 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn3 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn2 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn1 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        
        # Final classifier
        self.classifier = nn.Conv2d(self.fpn_channels * 4, num_classes, kernel_size=1)

    def conv_layer(self, in_channels, out_channels, kernel_size, padding):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        ]
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Extract features from backbone
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # Build FPN
        # Top-down pathway
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + self.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + self.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + self.interpolate(p2, scale_factor=2, mode='nearest')
        
        # Apply final convolutions
        p1 = self.fpn1(p1)
        p2 = self.fpn2(p2)
        p3 = self.fpn3(p3)
        p4 = self.fpn4(p4)
        
        # Upsample all to the p1 resolution using fixed scale factors (static shapes help torch.compile)
        # p1 : 1/4 input size, p2 : 1/8, p3 : 1/16, p4 : 1/32  -> scale factors: 2, 4, 8 respectively
        p2_up = self.interpolate(p2, scale_factor=2, mode='bilinear')
        p3_up = self.interpolate(p3, scale_factor=4, mode='bilinear')
        p4_up = self.interpolate(p4, scale_factor=8, mode='bilinear')
        
        # Combine features
        fused = torch.cat([p1, p2_up, p3_up, p4_up], dim=1)
        
        # Final classification
        out = self.classifier(fused)
        
        # Upsample to input size
        out = self.interpolate(out, scale_factor=4, mode='bilinear')
        
        return out

    def interpolate(self, x, size=None, scale_factor=None, mode='bilinear'):
        align_corners = False if mode == 'bilinear' else None
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class HyperbolicFPN(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None, use_batch_norm=True, use_mobius_addition=True):
        super().__init__()
        self.manifold = CustomLorentz(k=1.0, learnable=False)
        self.backbone = Lorentz_resnet18_wrapper(manifold=self.manifold)
        
        self.use_batch_norm = use_batch_norm
        self.use_mobius_addition = use_mobius_addition
        
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            weights = checkpoint['model']
            self.backbone.load_state_dict(weights, strict=False)

        self.feature_channels = [65, 129, 257, 513]
        self.fpn_channels = 257
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral4 = self.conv_layer(self.feature_channels[3], self.fpn_channels, kernel_size=1)
        self.lateral3 = self.conv_layer(self.feature_channels[2], self.fpn_channels, kernel_size=1)
        self.lateral2 = self.conv_layer(self.feature_channels[1], self.fpn_channels, kernel_size=1)
        self.lateral1 = self.conv_layer(self.feature_channels[0], self.fpn_channels, kernel_size=1)
        
        # Final convs after merging
        self.fpn4 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn3 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn2 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        self.fpn1 = self.conv_layer(self.fpn_channels, self.fpn_channels, kernel_size=3, padding=1)
        
        # Final classifier
        self.classifier = LorentzMLR(self.manifold, self.fpn_channels * 4, num_classes)
        
    def conv_layer(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        stride=1,
        padding=0
    ):
        return LConv2d_Block(
            manifold=self.manifold, 
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=True,
            activation=torch.relu,
            normalization="batch_norm" if self.use_batch_norm else None
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x, return_features=True)
        
        p4 = self.lateral4(c4)
        if self.use_mobius_addition:
            p3 = self.manifold.pt_addition(self.lateral3(c3), self.interpolate(p4, scale_factor=2, mode='nearest'))
            p2 = self.manifold.pt_addition(self.lateral2(c2), self.interpolate(p3, scale_factor=2, mode='nearest'))
            p1 = self.manifold.pt_addition(self.lateral1(c1), self.interpolate(p2, scale_factor=2, mode='nearest'))
        else:
            p3 = self.lateral3(c3) + self.interpolate(p4, scale_factor=2, mode='nearest')
            p2 = self.lateral2(c2) + self.interpolate(p3, scale_factor=2, mode='nearest')
            p1 = self.lateral1(c1) + self.interpolate(p2, scale_factor=2, mode='nearest')
        
        p1 = self.fpn1(p1)
        p2 = self.fpn2(p2)
        p3 = self.fpn3(p3)
        p4 = self.fpn4(p4)
        
        # p1 : 1/4 input ; p2 : 1/8 ; p3 : 1/16 ; p4 : 1/32  --> scale factors: 2,4,8 (avoid dynamic shapes)
        p2 = self.interpolate(p2, scale_factor=2, mode='hyperbolic')
        p3 = self.interpolate(p3, scale_factor=4, mode='hyperbolic')
        p4 = self.interpolate(p4, scale_factor=8, mode='hyperbolic')

        fused = torch.cat([p1, p2, p3, p4], dim=-1)
        
        out = self.classifier(fused)
        out = out.permute(0,3,1,2)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out

    def interpolate(self, x, size=None, scale_factor=None, mode='hyperbolic'):
        B, H, W, C = x.shape
        if size is not None and size[0] == H and size[1] == W:
            return x

        if mode == 'hyperbolic':
            return self._hyperbolic_interp(x, size, scale_factor)
        elif mode == 'nearest':
            return self._nearest_interp(x, size, scale_factor)
        else:
            raise ValueError(f"Invalid interpolation mode: {mode}")
    
    def _nearest_interp(self, x, size=None, scale_factor=None):
        B, H, W, C = x.shape
        if size is not None:
            scale_factor = (int(size[0]/H), int(size[1]/W))
            
        if scale_factor is not None and isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor)
        
        x = x.view(B, H, 1, W, 1, C)
        x = x.expand(B, H, scale_factor[0], W, scale_factor[1], C)
        x = x.reshape(B, H * scale_factor[0], W * scale_factor[1], C)
        return x

    def _hyperbolic_interp(self, x, size=None, scale_factor=None):
        """Performs bilinear up/down-sampling in the tangent space at the origin.

        Steps:
        1.  Map x \in H to the tangent space at the origin using logmap_0.
        2.  Apply ordinary bilinear interpolation on that Euclidean tangent space.
        3.  Map the result back to the manifold with expmap_0.

        This preserves the manifold constraint exactly and approximates
        geodesic interpolation much better than the previous coordinate-wise
        variant. It is still an approximation because the tangent space is
        taken at the origin for all pixels, but it is unbiased and keeps the
        output on H.
        """

        # x is expected in BHWC format (batch, height, width, channels)
        # 1. map to tangent space
        u = self.manifold.logmap0(x)  # BHWC

        # 2. permute to B C H W for torch.interpolate
        u = u.permute(0, 3, 1, 2)

        if size is not None:
            u_interp = F.interpolate(u, size=size, mode='bilinear', align_corners=False)
        else:
            u_interp = F.interpolate(u, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        # 3. back to BHWC
        u_interp = u_interp.permute(0, 2, 3, 1)

        # 4. map back to manifold
        return self.manifold.expmap0(u_interp)


def profile_model(model, input_tensor, model_name="Model", use_amp=False):
    print(f"--- Profiling {model_name} ---")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move model and input to device
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f}M")
    
    # Calculate parameter memory usage
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"Parameter memory: {param_memory/1e6:.2f}MB")
    
    # --------------------------- Warm-up & Memory reset ---------------------------
    # Perform a few forward passes BEFORE starting the profiler so that:
    #   • torch.compile has time to compile and autotune kernels
    #   • cudagraph capture (used by "reduce-overhead" mode) is completed
    #   • subsequent profiling measures stable inference latency only
    n_warmup = 3
    with torch.no_grad():
        for _ in range(n_warmup):
            with torch.cuda.amp.autocast(enabled=use_amp):
                _ = model(input_tensor)

    # GPU memory before forward pass (after warm-up)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gpu_memory_before = torch.cuda.memory_allocated() / 1e6
        print(f"GPU memory before forward pass: {gpu_memory_before:.2f}MB")

    # Memory and FLOPs profiling
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(input_tensor)
    
    # GPU memory after forward pass
    if torch.cuda.is_available():
        gpu_memory_after = torch.cuda.memory_allocated() / 1e6
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"GPU memory after forward pass: {gpu_memory_after:.2f}MB")
        print(f"GPU memory peak: {gpu_memory_peak:.2f}MB")
        print(f"GPU memory used during forward pass: {gpu_memory_after - gpu_memory_before:.2f}MB")

    print(f"\n--- Profiler Results ({'CUDA' if torch.cuda.is_available() else 'CPU'}) ---")
    print("--- Memory Usage ---")
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
    else:
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
    
    print("\n--- Time Usage ---")
    if torch.cuda.is_available():
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    print("-" * 50)


if __name__ == '__main__':
    # torch._dynamo.config.verbose = True
    # torch._inductor.config.debug = True
    
    
    # Common dummy input (Cityscapes-like resolution)
    dummy_input = torch.randn(4, 3, 224, 224)

    # 1) Baseline Euclidean FPN (no compilation, FP32)
    fpn_baseline = FPN(backbone='resnet18', num_classes=19)
    profile_model(fpn_baseline, dummy_input.clone(), model_name="FPN (ResNet-18, baseline)", use_amp=False)

    print("\n" * 2)

    # 2) Baseline Hyperbolic FPN (no compilation, FP32)
    hfpn_baseline = HyperbolicFPN(num_classes=19)
    profile_model(hfpn_baseline, dummy_input.clone(), model_name="HyperbolicFPN (Lorentz ResNet-18, baseline)", use_amp=False)
    
    # 3) Compiled + AMP Euclidean FPN (use reduce-overhead mode for faster compile)
    fpn_compiled = torch.compile(
        FPN(backbone='resnet18', num_classes=19),
        mode="reduce-overhead"
    )
    profile_model(fpn_compiled, dummy_input.clone(), model_name="FPN (ResNet-18, compiled)", use_amp=True)

    print("\n" * 2)

    # 4) Compiled + AMP Hyperbolic FPN
    hfpn_compiled = torch.compile(
        HyperbolicFPN(num_classes=19),
        mode="reduce-overhead"
    )
    profile_model(hfpn_compiled, dummy_input.clone(), model_name="HyperbolicFPN (Lorentz ResNet-18, compiled)", use_amp=True)