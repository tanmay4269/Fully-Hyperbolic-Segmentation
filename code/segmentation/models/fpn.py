import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Upsample all to same size and concatenate
        _, _, h, w = p1.shape
        p2_up = self.interpolate(p2, size=(h, w), mode='bilinear')
        p3_up = self.interpolate(p3, size=(h, w), mode='bilinear')
        p4_up = self.interpolate(p4, size=(h, w), mode='bilinear')
        
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
        
        _, h, w, _ = p1.shape
        p2 = self.interpolate(p2, size=(h, w), mode='hyperbolic')
        p3 = self.interpolate(p3, size=(h, w), mode='hyperbolic')
        p4 = self.interpolate(p4, size=(h, w), mode='hyperbolic')

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