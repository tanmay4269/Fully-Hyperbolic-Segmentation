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

from segmentation.fpn import FPN, HyperbolicFPN
from lib.geoopt.optim import RiemannianAdam, RiemannianSGD


class VOCDatasetWrapper:
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        self.dataset = VOCSegmentation(root=root, image_set=image_set, download=False, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return img, mask.squeeze(0).long()

class Trainer:
    def __init__(self, model, device, num_classes=21, use_hyperbolic=False):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.use_hyperbolic = use_hyperbolic
        
        if self.use_hyperbolic:
            self.optimizer = RiemannianAdam(model.parameters(), lr=5e-5, weight_decay=1e-5, stabilize=1)
        else:
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
    # model = FPN(backbone='resnet18', num_classes=21, pretrained=False)
    # trainer = Trainer(model, device, use_hyperbolic=False)
    
    model = HyperbolicFPN(num_classes=21)
    trainer = Trainer(model, device, use_hyperbolic=True)

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