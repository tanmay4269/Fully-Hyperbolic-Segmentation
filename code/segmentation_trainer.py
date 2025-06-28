import os
import argparse
from datetime import datetime
import logging
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, PolynomialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from segmentation.datasets.pascal_voc import VOCDataset
from segmentation.datasets.cityscapes import CityscapesDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchmetrics.classification import MulticlassJaccardIndex

from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from segmentation.models.fpn import FPN, HyperbolicFPN
from segmentation.models.erfnet import ERFNet 

import wandb
import optuna

from tqdm import tqdm


def dice_loss(pred, target, num_classes, ignore_index=255):
    """
    Calculates the multi-class Dice loss.
    """
    pred_softmax = F.softmax(pred, dim=1)
    
    # Create a mask for valid pixels
    valid_mask = target != ignore_index
    
    # Mask out ignored pixels from target for one-hot encoding
    masked_target = target.clone()
    masked_target[~valid_mask] = 0
    
    target_one_hot = F.one_hot(masked_target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Mask out ignored pixels from predictions and one-hot target
    valid_mask = valid_mask.unsqueeze(1)
    pred_softmax = pred_softmax * valid_mask
    target_one_hot = target_one_hot * valid_mask
    
    dims = (0, 2, 3)
    intersection = torch.sum(pred_softmax * target_one_hot, dims)
    cardinality = torch.sum(pred_softmax + target_one_hot, dims)
    
    dice_score = (2. * intersection + 1e-7) / (cardinality + 1e-7)
    return 1. - dice_score.mean()


def get_args():
    parser = argparse.ArgumentParser(description='Training script for segmentation models')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='cityscapes',
                      choices=['pascal-voc', 'cityscapes'],
                      help='Dataset to use for training')
    parser.add_argument('--data-root', type=str, default='data/cityscapes',
                      help='Root directory for dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 256],
                      help='Input image size (height, width)')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of workers for data loading')

    # Run Management
    parser.add_argument('--run-name', type=str, default=None,
                        help="A name for this run. If not provided, a name will be generated.")
    parser.add_argument('--log-dir', type=str, default='logs',
                        help="Base directory for logs and checkpoints.")
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    parser.add_argument('--save-interval', type=int, default=10,
                        help="Save a checkpoint every N epochs.")

    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with limited data and epochs')
    parser.add_argument('--save-model', action='store_true',
                      help='Save the best model during training')
    
    # Model parameters
    parser.add_argument('--manifold', type=str, default='hyperbolic', 
                        choices=['euclidean', 'hyperbolic'],
                        help='Type of manifold that the model is defined on')
    parser.add_argument('--use-batch-norm', action='store_true',
                        help='Use batch normalization in the model')
    parser.add_argument('--use-mobius-addition', action='store_true',
                        help='Use Mobius addition in the model')
    parser.add_argument('--num-classes', type=int, default=21,
                      help='Number of classes for segmentation')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      help='Backbone architecture for FPN')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights for the backbone')
    parser.add_argument('--pretrained-checkpoint-path', type=str, default=None)
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=200,
                      help='Number of training epochs')
    
    parser.add_argument('--subsample-percentage', type=float, default=100.0,
                        help='Percentage of dataset to use for training (0-100). Subsampling is class-balanced.')
    
    parser.add_argument('--optimizer', type=str, default='sgd',
                      choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Learning rate')
    parser.add_argument('--backbone-lr-factor', type=float, default=0.1,
                      help='Learning rate factor for the backbone')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay for optimizer')
    
    parser.add_argument('--use-amp', action='store_true',
                        help='Use Automatic Mixed Precision (AMP) for training.')
    
    parser.add_argument('--dice-weight', type=float, default=0.5,
                        help='Weight for Dice loss in the total loss function.')
    
    parser.add_argument('--lr-scheduler', type=str, default='multistep',
                        choices=['multistep', 'poly', 'reduce-on-plateau', 'none'],
                        help="Type of learning rate scheduler to use.")
    
    # MultiStepLR parameters
    parser.add_argument('--scheduler-multistep-milestones', default=[40, 100, 125], type=int, nargs="+",
                        help="Milestones for MultiStepLR scheduler.")
    parser.add_argument('--scheduler-multistep-gamma', default=0.2, type=float,
                        help="Gamma parameter for MultiStepLR scheduler.")

    # PolynomialLR parameters
    parser.add_argument('--scheduler-poly-power', type=float, default=0.9,
                        help="Power for PolynomialLR scheduler.")

    # ReduceLROnPlateau parameters
    parser.add_argument('--scheduler-rop-patience', type=int, default=10,
                        help="Patience for ReduceLROnPlateau scheduler.")
    parser.add_argument('--scheduler-rop-factor', type=float, default=0.1,
                        help="Factor for ReduceLROnPlateau scheduler.")
    
    # Wandb parameters
    parser.add_argument('--use-wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='fully-hyperbolic-segmentation',
                      help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                      help='Weights & Biases entity (username or team name)')
    parser.add_argument('--wandb-name', type=str, default=None,
                      help='Name of the run (optional)')
    
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
    
    if args.subsample_percentage < 0 or args.subsample_percentage > 100:
        raise ValueError("--subsample-percentage must be between 0 and 100.")
    return args
        

class Trainer:
    def __init__(self, model, device, args, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.num_classes = args.num_classes
        self.use_hyperbolic = args.manifold == 'hyperbolic'
        self.debug = args.debug
        self.use_wandb = args.use_wandb
        self.dice_weight = args.dice_weight
        self.class_weights = class_weights
        self.use_amp = args.use_amp and self.device.type == 'cuda'
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
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
            if args.optimizer == 'adam':
                self.optimizer = RiemannianAdam(param_groups, weight_decay=args.weight_decay, stabilize=1)
            elif args.optimizer == 'sgd':
                self.optimizer = RiemannianSGD(param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True, stabilize=1)
        else:
            if args.optimizer == 'adam':
                self.optimizer = Adam(param_groups, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                self.optimizer = SGD(param_groups, weight_decay=args.weight_decay)
        
        self.lr_scheduler = None
        if args.lr_scheduler == 'multistep':
            self.lr_scheduler = MultiStepLR(
                self.optimizer, 
                milestones=args.scheduler_multistep_milestones, 
                gamma=args.scheduler_multistep_gamma
            )
        elif args.lr_scheduler == 'poly':
            self.lr_scheduler = PolynomialLR(
                self.optimizer,
                total_iters=args.num_epochs,
                power=args.scheduler_poly_power
            )
        elif args.lr_scheduler == 'reduce-on-plateau':
            self.lr_scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=args.scheduler_rop_factor,
                patience=args.scheduler_rop_patience,
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

    def compute_loss(self, outputs, masks):
        """
        Computes the total loss, optionally combining Cross-Entropy and Dice loss.
        """
        ce_loss = F.cross_entropy(outputs, masks, weight=self.class_weights, ignore_index=255)
        
        if self.dice_weight > 0:
            dl = dice_loss(outputs, masks, self.num_classes, ignore_index=255)
            loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dl
        else:
            loss = ce_loss
            
        return loss

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        self.train_iou.reset()
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.compute_loss(outputs, masks)
            
            self.train_iou.update(outputs.argmax(dim=1), masks)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            if self.use_wandb and batch_idx % 10 == 0:  
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
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.compute_loss(outputs, masks)
                
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
    # Setup run directory and logging
    if not args.run_name:
        run_name_parts = [datetime.now().strftime("%Y-%m-%d_%H-%M-%S")]
        if args.wandb_name:
            run_name_parts.append(args.wandb_name)
        args.run_name = '_'.join(run_name_parts)
    
    args.run_dir = os.path.join(args.log_dir, args.run_name)
    args.checkpoint_dir = os.path.join(args.run_dir, 'checkpoints')
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Setup logging to file and console
    log_file = os.path.join(args.run_dir, 'run.log')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s')) # Cleaner console output
    logger.addHandler(console_handler)
    
    logging.info("Starting new run...")
    logging.info("Arguments:")
    logging.info("-" * 40)
    for arg, value in vars(args).items():
        logging.info(f"{arg:20s}: {value}")
    logging.info("-" * 40)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if args.use_amp and device.type == 'cuda':
        logging.info("Using Automatic Mixed Precision (AMP).")
    logging.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    # Set number of classes based on dataset
    if args.dataset == 'cityscapes':
        args.num_classes = 19
    elif args.dataset == 'pascal-voc':
        args.num_classes = 21

    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )

    # Create checkpoint directory if needed
    if args.save_model:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Class balancing is now handled inside the dataset class for Cityscapes.
    # For other datasets, this needs to be implemented in their respective classes.
    if args.dataset == 'pascal-voc':
        logging.info("Class balancing for Pascal VOC is not implemented in the dataset class yet.")

    # Define albumentations transforms for training
    if args.debug:
        train_transform = A.Compose([
            A.Resize(height=args.img_size[0], width=args.img_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        train_transform = A.Compose([
            A.Resize(height=args.img_size[0], width=args.img_size[1]),
            A.Sequential([
                A.RandomScale(scale_limit=(-0.5, 1.0)),
                A.PadIfNeeded(min_height=args.img_size[0], min_width=args.img_size[1]),
                A.RandomCrop(height=args.img_size[0], width=args.img_size[1]),
            ]),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
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

    if args.dataset == 'pascal-voc':
        train_dataset = VOCDataset(
            args.data_root,
            image_set='train',
            albumentation_transform=train_transform,
        )

        val_dataset = VOCDataset(
            args.data_root,
            image_set='val',
            albumentation_transform=val_transform,
        )
    elif args.dataset == 'cityscapes':
        train_dataset = CityscapesDataset(
            root=args.data_root,
            split='train',
            albumentation_transform=train_transform,
            subsample_percentage=args.subsample_percentage,
        )
        val_dataset = CityscapesDataset(
            root=args.data_root,
            split='val',
            albumentation_transform=val_transform,
            subsample_percentage=args.subsample_percentage,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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

    # Get class_weights from dataset object and move to device
    class_weights = None
    if hasattr(train_dataset, 'class_weights') and train_dataset.class_weights is not None:
        class_weights= train_dataset.class_weights.to(device)

    # Initialize model based on arguments
    if args.manifold == 'hyperbolic':
        model = HyperbolicFPN(
            num_classes=args.num_classes,
            checkpoint_path=args.pretrained_checkpoint_path,
            use_batch_norm=args.use_batch_norm,
            use_mobius_addition=args.use_mobius_addition
        )
    else:
        # model = ERFNet(
        #     num_classes=args.num_classes,
        # )

        model = FPN(
            backbone=args.backbone,
            num_classes=args.num_classes, 
            pretrained=args.pretrained,
            use_batch_norm=args.use_batch_norm
        )
        
    
    trainer = Trainer(model, device, args, class_weights=class_weights)

    # Resuming from checkpoint
    start_epoch = 0
    best_val_iou = 0.0
    if args.resume_checkpoint:
        if os.path.isfile(args.resume_checkpoint):
            logging.info(f"Loading checkpoint '{args.resume_checkpoint}'")
            checkpoint = torch.load(args.resume_checkpoint, map_location=device)
            
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if trainer.lr_scheduler and 'scheduler_state_dict' in checkpoint:
                trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'val_iou' in checkpoint:
                 best_val_iou = checkpoint['val_iou']
            
            logging.info(f"Resumed training from epoch {start_epoch}. Previous best mIoU: {best_val_iou:.4f}")
        else:
            logging.error(f"No checkpoint found at '{args.resume_checkpoint}'")

    # Training loop
    num_epochs = args.num_epochs
    
    for epoch in range(start_epoch, num_epochs):
        logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_iou = trainer.train_epoch(train_loader)
        val_loss, val_iou = trainer.validate(val_loader)
        
        logging.info(f"Training Loss: {train_loss:.4f}, Training mIoU: {train_iou:.4f}")
        logging.info(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_iou:.4f}")
        
        if trainer.lr_scheduler is not None:
            if isinstance(trainer.lr_scheduler, ReduceLROnPlateau):
                trainer.lr_scheduler.step(val_iou)
            else:
                trainer.lr_scheduler.step()
        
        # Pruning if Optuna is enabled
        if trial is not None and train_iou > 0:
            iou_ratio = val_iou / train_iou
            trial.report(val_iou, epoch)
            
            if iou_ratio < args.prune_threshold and epoch >= args.prune_patience:
                logging.info(f"Trial pruned at epoch {epoch+1} with val_iou/train_iou = {iou_ratio:.4f}")
                if args.use_wandb:
                    wandb.finish()
                raise optuna.TrialPruned()
        
        # Log metrics to wandb
        if args.use_wandb:
            log_data = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/mIoU': train_iou,
                'val/loss': val_loss,
                'val/mIoU': val_iou,
                'lr/backbone': trainer.optimizer.param_groups[0]['lr'],
                'lr/head': trainer.optimizer.param_groups[1]['lr'],
            }
            if train_iou > 0:
                log_data['g-score'] = val_iou / train_iou
            wandb.log(log_data)

        if args.save_model and (epoch + 1) % args.save_interval == 0:
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
            if trainer.lr_scheduler:
                checkpoint['scheduler_state_dict'] = trainer.lr_scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved model checkpoint to {checkpoint_path}")
            
            # Log model checkpoint to wandb
            if args.use_wandb:
                wandb.save(checkpoint_path)
        
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
            if trainer.lr_scheduler:
                checkpoint['scheduler_state_dict'] = trainer.lr_scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved best model checkpoint to {checkpoint_path} (best IoU)")
            
            # Log best model to wandb
            if args.use_wandb:
                wandb.save(checkpoint_path)
                wandb.run.summary['best_val_iou'] = best_val_iou
                wandb.run.summary['best_epoch'] = epoch + 1

    # Close wandb run
    if args.use_wandb:
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
            if trial_args.use_wandb:
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