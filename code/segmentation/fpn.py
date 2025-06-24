import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from lib.models.resnet import Lorentz_resnet18_wrapper
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzConv2d, LorentzMLR
from lib.lorentz.blocks.layer_blocks import LConv2d_Block


class FPN(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=1, pretrained=False):
        super().__init__()
        
        # Load backbone and get feature channels
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=pretrained)
            self.feature_channels = [64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=pretrained)
            self.feature_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Backbone {backbone} not supported")
        
        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # FPN components
        self.fpn_channels = 256
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral4 = nn.Conv2d(self.feature_channels[3], self.fpn_channels, 1)
        self.lateral3 = nn.Conv2d(self.feature_channels[2], self.fpn_channels, 1)
        self.lateral2 = nn.Conv2d(self.feature_channels[1], self.fpn_channels, 1)
        self.lateral1 = nn.Conv2d(self.feature_channels[0], self.fpn_channels, 1)
        
        # Final convs after merging
        self.fpn4 = nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn3 = nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn2 = nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn1 = nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1)
        
        # Final classifier
        self.classifier = nn.Conv2d(self.fpn_channels, num_classes, 1)

    def forward(self, x):
        # Extract features from backbone
        features = []
        for i, module in enumerate(self.backbone):
            x = module(x)
            if i in [4, 5, 6, 7]:  # layer1, layer2, layer3, layer4
                features.append(x)
        
        c1, c2, c3, c4 = features
        
        # Build FPN
        # Top-down pathway
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
        
        # Apply final convolutions
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)
        p1 = self.fpn1(p1)
        
        # Upsample all to same size and add
        _, _, h, w = p1.shape
        p4_up = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        
        # Combine features
        fused = p1 + p2_up + p3_up + p4_up
        
        # Final classification
        out = self.classifier(fused)
        
        # Upsample to input size
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out


class HyperbolicFPN(nn.Module):
    def __init__(self, num_classes, checkpoint_path=None):
        super().__init__()
        self.manifold = CustomLorentz(k=1.0, learnable=False)
        self.backbone = Lorentz_resnet18_wrapper(manifold=self.manifold)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            weights = checkpoint['model']
            # self.backbone.load_state_dict(weights, strict=True)
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
        self.classifier = LorentzMLR(self.manifold, self.fpn_channels, num_classes)
        
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
            normalization="batch_norm"
        )

    def forward(self, x):
        c1, c2, c3, c4 = self.backbone(x, return_features=True)
        
        p4 = self.lateral4(c4)
        p3 = self.manifold.pt_addition(self.lateral3(c3), self.interpolate(p4, scale_factor=2, method='hyperbolic'))
        p2 = self.manifold.pt_addition(self.lateral2(c2), self.interpolate(p3, scale_factor=2, method='hyperbolic'))
        p1 = self.manifold.pt_addition(self.lateral1(c1), self.interpolate(p2, scale_factor=2, method='hyperbolic'))
        
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)
        p1 = self.fpn1(p1)
        
        _, h, w, _ = p1.shape
        p4 = self.interpolate(p4, size=(h, w), method='hyperbolic')
        p3 = self.interpolate(p3, size=(h, w), method='hyperbolic')
        p2 = self.interpolate(p2, size=(h, w), method='hyperbolic')
        
        fused = self.manifold.pt_addition(p1, self.manifold.pt_addition(p2, self.manifold.pt_addition(p3, p4)))
        
        out = self.classifier(fused)
        out = out.permute(0,3,1,2)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out

    def interpolate(self, x, size=None, scale_factor=None, method='hyperbolic'):
        B, H, W, C = x.shape
        if size is not None and size[0] == H and size[1] == W:
            return x

        if method == 'hyperbolic':
            return self._hyperbolic_interp(x, size, scale_factor)
        elif method == 'nearest':
            return self._nearest_interp(x, size, scale_factor)
        else:
            raise ValueError(f"Invalid interpolation method: {method}")
    
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