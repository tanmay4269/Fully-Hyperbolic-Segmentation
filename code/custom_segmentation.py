import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse

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


def get_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation models')
    # Model parameters
    parser.add_argument('--model-type', type=str, default='hyperbolic', choices=['standard', 'hyperbolic'],
                      help='Type of model to use (standard or hyperbolic)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      help='Backbone architecture for FPN')
    parser.add_argument('--num-classes', type=int, default=21,
                      help='Number of classes for segmentation')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay for optimizer')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='data/pascal_voc',
                      help='Root directory for dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256],
                      help='Input image size (height, width)')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of workers for data loading')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with limited data and epochs')
    parser.add_argument('--save-model', action='store_true',
                      help='Save the best model during training')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                      help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    return args

class VOCDatasetWrapper:
    def __init__(self, root, image_set='train', transform=None, target_transform=None):
        self.dataset = VOCSegmentation(root=root, image_set=image_set, download=False, transform=transform, target_transform=target_transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return img, mask.squeeze(0).long()

class Trainer:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.num_classes = args.num_classes
        self.use_hyperbolic = args.model_type == 'hyperbolic'
        self.debug = args.debug
        
        if self.use_hyperbolic:
            self.optimizer = RiemannianAdam(model.parameters(), 
                                          lr=args.lr, 
                                          weight_decay=args.weight_decay, 
                                          stabilize=1)
        else:
            self.optimizer = Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
        
        # Initialize IoU metric
        self.train_iou = MulticlassJaccardIndex(
            num_classes=args.num_classes,
            ignore_index=255
        ).to(device)
        self.val_iou = MulticlassJaccardIndex(
            num_classes=args.num_classes,
            ignore_index=255
        ).to(device)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        self.train_iou.reset()
        
        for images, masks in tqdm(dataloader, desc="Training", disable=not self.debug):
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
            
            if self.debug:
                break
        
        mean_iou = self.train_iou.compute()
        if self.debug:
            return total_loss, mean_iou.item()
        return total_loss / len(dataloader), mean_iou.item()

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        self.val_iou.reset()
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validation", disable=not self.debug):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, masks, ignore_index=255)
                
                # Calculate IoU
                self.val_iou.update(outputs.argmax(dim=1), masks)
                
                total_loss += loss.item()
                
                if self.debug:
                    break
        
        mean_iou = self.val_iou.compute()
        if self.debug:
            return total_loss, mean_iou.item()
        return total_loss / len(dataloader), mean_iou.item()

def main():
    args = get_args()
    print("Arguments:")
    print("-" * 40)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("-" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    # Create checkpoint directory if needed
    if args.save_model:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])

    train_dataset = VOCDatasetWrapper(
        args.data_root,
        image_set='train',
        transform=train_transform,
        target_transform=target_transform,
    )

    val_dataset = VOCDatasetWrapper(
        args.data_root,
        image_set='val',
        transform=val_transform,
        target_transform=target_transform,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=not args.debug,  # Don't shuffle in debug mode
        num_workers=args.num_workers if not args.debug else 0,  # Use 0 workers in debug mode
        pin_memory=True
    )
    
    if args.debug:
        val_loader = train_loader
    else:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    # Initialize model based on arguments
    if args.model_type == 'hyperbolic':
        model = HyperbolicFPN(num_classes=args.num_classes)
    else:
        model = FPN(backbone=args.backbone, 
                   num_classes=args.num_classes, 
                   pretrained=not args.debug)  # Don't load pretrained in debug mode
    
    trainer = Trainer(model, device, args)

    # Training loop
    num_epochs = args.num_epochs
        
    best_val_loss = float('inf')
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_iou = trainer.train_epoch(train_loader)
        val_loss, val_iou = trainer.validate(val_loader)
        
        print(f"Training Loss: {train_loss:.4f}, Training mIoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_iou:.4f}")
        
        # Save best model based on IoU
        if args.save_model and val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'args': args,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path} (best IoU)")

if __name__ == "__main__":
    main() 