import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import torchvision.transforms.functional as TF
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import os
from torchmetrics.classification import MulticlassJaccardIndex

import segmentation_models_pytorch as smp

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class CustomSegmentation(nn.Module):
    def __init__(self, in_channels=3, num_classes=21):  # 21 classes for Pascal VOC
        super(CustomSegmentation, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)
        
        # Decoder
        self.dec4 = ConvBlock(1024 + 512, 512)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        
        return self.final(d1)

class VOCDatasetWrapper:
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        self.dataset = VOCSegmentation(root=root, image_set=image_set, download=False, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return img, mask.squeeze(0).long()

class Trainer:
    def __init__(self, model, device, num_classes=21):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.optimizer = Adam(model.parameters(), lr=1e-4)
        
        # Initialize IoU metric
        self.train_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=255
        ).to(device)
        self.val_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=255
        ).to(device)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        self.train_iou.reset()
        
        for images, masks in tqdm(dataloader, desc="Training", disable=True):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, masks, ignore_index=255)
            
            # Calculate IoU
            self.train_iou.update(outputs.argmax(dim=1), masks)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            break
        
        mean_iou = self.train_iou.compute()
        return total_loss, mean_iou.item()
        # return total_loss / len(dataloader), mean_iou.item()

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        self.val_iou.reset()
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validation", disable=True):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, masks, ignore_index=255)
                
                # Calculate IoU
                self.val_iou.update(outputs.argmax(dim=1), masks)
                
                total_loss += loss.item()
                
                break
        
        mean_iou = self.val_iou.compute()
        return total_loss, mean_iou.item()
        # return total_loss / len(dataloader), mean_iou.item()

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    img_size = (256, 256)  

    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    train_dataset = VOCDatasetWrapper(
        'data/pascal_voc',
        image_set='train',
        transform=train_transform,
        target_transform=target_transform,
    )

    val_dataset = VOCDatasetWrapper(
        'data/pascal_voc',
        image_set='val',
        transform=val_transform,
        target_transform=target_transform,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    val_loader = train_loader
    # val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model and trainer
    # model = CustomSegmentation()
    # model = smp.FPN(
    #     encoder_name="resnet18",
    #     encoder_weights=None,
    #     # encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=21,
    # )
    model = LSeg(
        enc_layers=5,
        dec_layers=5,
        num_classes=21,
        initial_filters=64,
        learn_curvature=False,
    )
    trainer = Trainer(model, device)

    # Training loop
    num_epochs = 200
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_iou = trainer.train_epoch(train_loader)
        val_loss, val_iou = trainer.validate(val_loader)
        
        print(f"Training Loss: {train_loss:.4f}, Training mIoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_iou:.4f}")
        
        # Save best model based on IoU
        # if val_iou > best_val_iou:
        #     best_val_iou = val_iou
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': trainer.optimizer.state_dict(),
        #         'val_loss': val_loss,
        #         'val_iou': val_iou,
        #     }, 'best_model.pth')
        #     print("Saved best model checkpoint (best IoU)")

if __name__ == "__main__":
    main() 