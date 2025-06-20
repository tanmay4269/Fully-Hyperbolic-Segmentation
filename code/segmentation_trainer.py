import os
import argparse
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchmetrics.classification import MulticlassJaccardIndex

import seg_models 
from lib.geoopt.optim import RiemannianAdam 

import wandb
import optuna


def get_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation models')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='hyperbolic', 
                        choices=['standard', 'hyperbolic'],
                        help='Type of model to use (standard or hyperbolic)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      help='Backbone architecture for FPN')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the backbone')
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=None)
    parser.add_argument('--num-classes', type=int, default=21,
                      help='Number of classes for segmentation')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                      help='Learning rate')
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1,
                      help='Learning rate factor for the backbone')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay for optimizer')
    
    parser.add_argument('--use-lr-scheduler', action='store_true',
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr-scheduler-milestones', default=[40, 100, 125], type=int, nargs="+",
                        help="Milestones of LR scheduler.")
    parser.add_argument('--lr-scheduler-gamma', default=0.2, type=float,
                        help="Gamma parameter of LR scheduler.")
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
    parser.add_argument('--checkpoint-dir', type=str,
                      default=f'checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
                      help='Directory to save model checkpoints')
    
    # Wandb parameters
    parser.add_argument('--wandb-project', type=str, default='fully-hyperbolic-segmentation',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                      help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb-name', type=str, default=None,
                      help='Name of the run (optional)')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    
    # Optuna parameters
    parser.add_argument('--use-optuna', action='store_true',
                        help='Enable hyperparameter tuning with Optuna')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials for Optuna study')
    parser.add_argument('--prune-threshold', type=float, default=0.85,
                        help='Threshold for pruning Optuna study')
    parser.add_argument('--prune-patience', type=int, default=5,
                        help='Number of epochs to wait before pruning')
    
    args = parser.parse_args()
    return args

class VOCDatasetWrapper:
    def __init__(self, root, image_set='train', albumentation_transform=None):
        self.dataset = VOCSegmentation(root=root, image_set=image_set, download=False, transform=None, target_transform=None)
        self.albumentation_transform = albumentation_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        if self.albumentation_transform is None:
            return transforms.ToTensor()(img), torch.tensor(np.array(mask)).long()
        
        img_np = np.array(img)
        mask_np = np.array(mask)
        
        transformed = self.albumentation_transform(image=img_np, mask=mask_np)
        img_transformed = transformed['image']
        mask_transformed = transformed['mask']
        
        # Handle the case where mask_transformed might already be a tensor
        if isinstance(mask_transformed, torch.Tensor):
            return img_transformed, mask_transformed.long()
        else:
            return img_transformed, torch.from_numpy(mask_transformed).long()
        

class Trainer:
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.num_classes = args.num_classes
        self.use_hyperbolic = args.model_type == 'hyperbolic'
        self.debug = args.debug
        self.use_wandb = not args.no_wandb
        
        # Separate backbone and head parameters for different learning rates
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_factor},
            {'params': head_params, 'lr': args.lr}
        ]
        
        if self.use_hyperbolic:
            self.optimizer = RiemannianAdam(param_groups, weight_decay=args.weight_decay, stabilize=1)
        else:
            self.optimizer = Adam(param_groups, weight_decay=args.weight_decay)
        
        self.lr_scheduler = None
        if args.use_lr_scheduler:
            self.lr_scheduler = MultiStepLR(
                self.optimizer, 
                milestones=args.lr_scheduler_milestones, 
                gamma=args.lr_scheduler_gamma
            )
        
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
        
        for batch_idx, (images, masks) in enumerate(dataloader):
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
            
            # Log batch metrics
            if self.use_wandb and batch_idx % 10 == 0:  # Log every 10 batches
                wandb.log({
                    'batch/train_loss': loss.item(),
                })
            
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
            for images, masks in dataloader:
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

def run_training(args, trial=None):
    """
    Runs the training and validation loop for the segmentation model.
    
    Args:
        args: Command line arguments
        trial: Optional Optuna trial object for hyperparameter optimization
    
    Returns:
        best_val_iou: Best validation IoU achieved during training
    """
    print("Arguments:")
    print("-" * 40)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("-" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )

    # Create checkpoint directory if needed
    if args.save_model:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Define albumentations transforms for training
    train_transform = A.Compose([
        A.OneOf([
            A.Resize(height=args.img_size[0], width=args.img_size[1]),
            A.Sequential([
                A.RandomScale(scale_limit=0.2),
                A.PadIfNeeded(min_height=args.img_size[0], min_width=args.img_size[1]),
                A.RandomCrop(height=args.img_size[0], width=args.img_size[1]),
            ])
        ], p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Define albumentations transforms for validation
    val_transform = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = VOCDatasetWrapper(
        args.data_root,
        image_set='train',
        albumentation_transform=train_transform,
    )

    val_dataset = VOCDatasetWrapper(
        args.data_root,
        image_set='val',
        albumentation_transform=val_transform,
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
        model = seg_models.HyperbolicFPN(
            num_classes=args.num_classes,
            checkpoint_path=args.pretrained_checkpoint_path
        )
    else:
        model = seg_models.FPN(
            backbone=args.backbone,
            num_classes=args.num_classes, 
            pretrained=args.pretrained
        )
        
    
    trainer = Trainer(model, device, args)

    # Training loop
    num_epochs = args.num_epochs
    best_val_iou = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_iou = trainer.train_epoch(train_loader)
        val_loss, val_iou = trainer.validate(val_loader)
        
        print(f"Training Loss: {train_loss:.4f}, Training mIoU: {train_iou:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_iou:.4f}")
        
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()
        
        # Pruning if Optuna is enabled
        if trial is not None and train_iou > 0:
            iou_ratio = val_iou / train_iou
            trial.report(val_iou, epoch)
            
            if iou_ratio < args.prune_threshold and epoch >= args.prune_patience:
                print(f"Trial pruned at epoch {epoch+1} with val_iou/train_iou = {iou_ratio:.4f}")
                raise optuna.TrialPruned()
        
        # Log metrics to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/mIoU': train_iou,
                'val/loss': val_loss,
                'val/mIoU': val_iou,
            })

        if args.save_model and (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'args': args,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
            
            # Log model checkpoint to wandb
            if not args.no_wandb:
                wandb.save(checkpoint_path)
        
        # Save best model based on IoU
        if args.save_model and val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'args': args,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path} (best IoU)")
            
            # Log best model to wandb
            if not args.no_wandb:
                wandb.save(checkpoint_path)
                wandb.run.summary['best_val_iou'] = best_val_iou
                wandb.run.summary['best_epoch'] = epoch + 1

    # Close wandb run
    if not args.no_wandb:
        wandb.finish()
    
    return best_val_iou

def main():
    """
    Main entry point for the script.
    Handles either a single training run or an Optuna hyperparameter study.
    """
    base_args = get_args()

    if base_args.use_optuna:
        def objective(trial):
            trial_args = argparse.Namespace(**vars(base_args))

            # trial_args.pretrained = trial.suggest_categorical('pretrained', [True, False])
            trial_args.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            trial_args.backbone_lr_factor = trial.suggest_float('backbone_lr_factor', 0.01, 0.5, log=True)
            trial_args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

            # Setting a unique name for wandb run if it's enabled
            if not trial_args.no_wandb:
                trial_args.wandb_name = f"trial-{trial.number}"

            print(f"\nStarting trial {trial.number} with params: {trial.params}")
            
            return run_training(trial_args, trial)

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=base_args.n_trials)

        print("Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial

        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        run_training(base_args)


if __name__ == "__main__":
    main() 