import os
from typing import Optional, Dict, Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models import resnet18, ResNet18_Weights
import torchmetrics


class UNetWithResNetBackbone(nn.Module):
    def __init__(self, n_classes: int = 21, backbone: str = 'resnet18'):
        super().__init__()
        
        # Load pre-trained ResNet backbone
        if backbone == 'resnet18':
            # self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.backbone = resnet18()
        
        # Encoder layers (using ResNet layers)
        self.encoder1 = nn.Sequential(self.backbone.conv1, 
                                    self.backbone.bn1,
                                    self.backbone.relu)  # 64 channels
        self.encoder2 = self.backbone.layer1  # 64 channels
        self.encoder3 = self.backbone.layer2  # 128 channels
        self.encoder4 = self.backbone.layer3  # 256 channels
        self.encoder5 = self.backbone.layer4  # 512 channels
        
        # Decoder layers
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256, 128)
        self.decoder2 = self._make_decoder_block(128, 64)
        self.decoder1 = self._make_decoder_block(64, 32)
        
        # Final classification layer
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder path with skip connections
        d4 = self.decoder4(e5) + e4
        d3 = self.decoder3(d4) + e3
        d2 = self.decoder2(d3) + e2
        d1 = self.decoder1(d2)
        
        # Final classification
        return self.final(d1)

class VOCSegmentationModule(pl.LightningModule):
    def __init__(
        self,
        data_dir: str = "data/pascal_voc",
        batch_size: int = 8,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        backbone: str = 'resnet18'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = UNetWithResNetBackbone(n_classes=21, backbone=backbone)
        
        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=21)
        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=21)
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.target_transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        y = y.squeeze(1).long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate IoU
        preds = torch.argmax(logits, dim=1)
        iou = self.train_iou(preds, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        y = y.squeeze(1).long()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Calculate IoU
        preds = torch.argmax(logits, dim=1)
        iou = self.val_iou(preds, y)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5 
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def train_dataloader(self) -> DataLoader:
        dataset = VOCSegmentation(
            root=self.hparams.data_dir,
            year='2012',
            image_set='train',
            download=True,
            transform=self.transform,
            target_transform=self.target_transform
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return train_loader
    
    def val_dataloader(self) -> DataLoader:
        dataset = VOCSegmentation(
            root=self.hparams.data_dir,
            year='2012',
            image_set='val',
            download=True,
            transform=self.transform,
            target_transform=self.target_transform
        )
        val_loader = DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        return val_loader
    
def main():
    torch.set_float32_matmul_precision('high')

    # Initialize model and trainer
    model = VOCSegmentationModule(
        data_dir="data/pascal_voc",
        batch_size=8,
        num_workers=4,
        learning_rate=1e-3,
        backbone='resnet18'
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='auto',
        devices=1,
        logger=pl.loggers.TensorBoardLogger('logs/', name='voc_segmentation'),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_iou',
                mode='max',
                save_top_k=3,
                filename='{epoch}-{val_iou:.2f}'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )
    
    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    main()
