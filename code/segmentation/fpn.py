import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101

from lib.models.resnet import Lorentz_resnet18
from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzConv2d, LorentzMLR
from lib.lorentz.blocks.layer_blocks import LFC_Block, LConv2d_Block, LTransposedConv2d_Block


class FPN(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=1, pretrained=True):
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
    def __init__(self, num_classes):
        super().__init__()
        self.manifold = CustomLorentz(k=1.0, learnable=False)

        self.backbone = Lorentz_resnet18(manifold=self.manifold)

        self.feature_channels = [65, 129, 257, 513]
        self.fpn_channels = 257
        
        
        # Lateral connections (1x1 conv to reduce channels)
        self.lateral4 = LorentzConv2d(self.manifold, self.feature_channels[3], self.fpn_channels, 1)
        self.lateral3 = LorentzConv2d(self.manifold, self.feature_channels[2], self.fpn_channels, 1)
        self.lateral2 = LorentzConv2d(self.manifold, self.feature_channels[1], self.fpn_channels, 1)
        self.lateral1 = LorentzConv2d(self.manifold, self.feature_channels[0], self.fpn_channels, 1)
        
        # Final convs after merging
        self.fpn4 = LorentzConv2d(self.manifold, self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn3 = LorentzConv2d(self.manifold, self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn2 = LorentzConv2d(self.manifold, self.fpn_channels, self.fpn_channels, 3, padding=1)
        self.fpn1 = LorentzConv2d(self.manifold, self.fpn_channels, self.fpn_channels, 3, padding=1)
        
        # Final classifier
        self.classifier = LorentzMLR(self.manifold, self.fpn_channels, num_classes)
    
    def upsample_bhwc(self, x, scale_factor=2):
        B, H, W, C = x.shape
        x = x.view(B, H, 1, W, 1, C)
        x = x.expand(B, H, scale_factor, W, scale_factor, C)
        x = x.reshape(B, H * scale_factor, W * scale_factor, C)
        return x

    def forward(self, x):
        # x = x.permute(0,2,3,1)
        # x = F.pad(x, pad=(1,0), mode="constant", value=0)
        # x = self.manifold.projx(x)

        # features = []
        # for i, module in enumerate(self.backbone):
        #     x = module(x)
        #     if i > 1:  # Out of 6 layers, skipping first two
        #         features.append(x)
        
        # c1, c2, c3, c4 = features
        c1, c2, c3, c4 = self.backbone(x, return_features=True)
        
        # Build FPN
        # Top-down pathway
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + self.upsample_bhwc(p4)
        p2 = self.lateral2(c2) + self.upsample_bhwc(p3)
        p1 = self.lateral1(c1) + self.upsample_bhwc(p2)
        
        p4 = self.fpn4(p4)
        p3 = self.fpn3(p3)
        p2 = self.fpn2(p2)
        p1 = self.fpn1(p1)
        
        p4 = p4.permute(0,3,1,2)
        p3 = p3.permute(0,3,1,2)
        p2 = p2.permute(0,3,1,2)
        p1 = p1.permute(0,3,1,2)
        
        # Upsample all to same size and add
        _, _, h, w = p1.shape
        p4_up = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False)
        
        # Combine features
        fused = p1 + p2_up + p3_up + p4_up
        fused = fused.permute(0,2,3,1)
        
        # Final classification
        out = self.classifier(fused)
        out = out.permute(0,3,1,2)
        
        # Upsample to input size
        # out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        
        return out