import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassAccuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2

import lib.models.resnet as resnet
from lib.geoopt.optim import RiemannianAdam 

import wandb
import optuna


def get_args():
    """
    Parses and returns the command line arguments.
    """
    parser = argparse.ArgumentParser(description='Training script for classification models')
    
    # Model parameters
    parser.add_argument('--model-type', type=str, default='hyperbolic', 
                        choices=['standard', 'hyperbolic'],
                        help='Type of model to use (standard or hyperbolic)')
    parser.add_argument('--backbone', type=str, default='resnet18',
                      help='Backbone architecture for ResNet')
    parser.add_argument('--num-classes', type=int, default=200,
                      help='Number of classes for classification (default: 200 for TinyImageNet)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=200,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay for optimizer')
    
    # Data parameters
    parser.add_argument('--data-root', type=str, default='data/tiny-imagenet-200',
                      help='Root directory for dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[64, 64],
                      help='Input image size (height, width)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Debug parameters
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with limited data and epochs')
    parser.add_argument('--save-model', action='store_true',
                      help='Save the best model during training')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_clf',
                      help='Directory to save model checkpoints')
    
    # Wandb parameters
    parser.add_argument('--wandb-project', type=str, default='fully-hyperbolic-classification',
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
    parser.add_argument('--prune-threshold', type=float, default=0.9,
                        help='Threshold for pruning Optuna study')
    parser.add_argument('--prune-patience', type=int, default=5,
                        help='Number of epochs to wait before pruning')
    
    args = parser.parse_args()
    return args
        

class ImageFolderWrapper:
    """
    A wrapper for ImageFolder that allows using albumentations transforms.

    Args:
        root (str): Root directory path.
        transform (callable, optional): An albumentations transform.
    """
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root=root)
        self.transform = transform
        self.loader = self.dataset.loader

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, target = self.dataset.samples[idx]
        img = self.loader(img_path)
        
        img_np = np.array(img)
        
        if self.transform:
            transformed = self.transform(image=img_np)
            img = transformed['image']
        
        return img, target
        

class Trainer:
    """
    A class to handle the training and validation of a model.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to run the training on.
        args (argparse.Namespace): The command line arguments.
    """
    def __init__(self, model, device, args):
        self.model = model.to(device)
        self.device = device
        self.num_classes = args.num_classes
        self.use_hyperbolic = args.model_type == 'hyperbolic'
        self.debug = args.debug
        self.use_wandb = not args.no_wandb
        
        if self.use_hyperbolic:
            self.optimizer = RiemannianAdam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
        else:
            self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # Initialize Accuracy metric
        self.train_acc = MulticlassAccuracy(
            num_classes=args.num_classes,
        ).to(device)
        self.val_acc = MulticlassAccuracy(
            num_classes=args.num_classes,
        ).to(device)

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): The DataLoader for the training data.
        
        Returns:
            A tuple containing the average training loss and accuracy.
        """
        self.model.train()
        total_loss = 0
        self.train_acc.reset()
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = F.cross_entropy(outputs, labels)
            
            self.train_acc.update(outputs.argmax(dim=1), labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if self.use_wandb and batch_idx % 20 == 0:
                wandb.log({'batch/train_loss': loss.item()})
            
            if self.debug:
                break
        
        mean_acc = self.train_acc.compute()
        if self.debug:
            return total_loss, mean_acc.item()
        return total_loss / len(dataloader), mean_acc.item()

    def validate(self, dataloader):
        """
        Runs validation on the given data.

        Args:
            dataloader (DataLoader): The DataLoader for the validation data.
        
        Returns:
            A tuple containing the average validation loss and accuracy.
        """
        self.model.eval()
        total_loss = 0
        self.val_acc.reset()
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = F.cross_entropy(outputs, labels)
                
                self.val_acc.update(outputs.argmax(dim=1), labels)
                
                total_loss += loss.item()
                
                if self.debug:
                    break
        
        mean_acc = self.val_acc.compute()
        if self.debug:
            return total_loss, mean_acc.item()
        return total_loss / len(dataloader), mean_acc.item()

def run_training(args, trial=None):
    """
    Runs the full training and validation loop.

    Args:
        args (argparse.Namespace): Command line arguments.
        trial (optuna.trial.Trial, optional): Optuna trial object. Defaults to None.

    Returns:
        float: The best validation accuracy achieved.
    """
    print("Arguments:")
    print("-" * 40)
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    print("-" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Debug mode: {'enabled' if args.debug else 'disabled'}")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )

    if args.save_model:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

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
    
    val_transform = A.Compose([
        A.Resize(height=args.img_size[0], width=args.img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = ImageFolderWrapper(
        root=os.path.join(args.data_root, 'train'),
        transform=train_transform,
    )
    val_dataset = ImageFolderWrapper(
        root=os.path.join(args.data_root, 'val'),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=not args.debug,
        num_workers=args.num_workers if not args.debug else 0,
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

    model_factory = {
        'resnet18': resnet.resnet18, 
        'resnet34': resnet.resnet34, 
        'resnet50': resnet.resnet50,
    }
    lorentz_model_factory = {
        'resnet18': resnet.Lorentz_resnet18, 
        'resnet34': resnet.Lorentz_resnet34, 
        'resnet50': resnet.Lorentz_resnet50,
    }
    
    model_kwargs = {'num_classes': args.num_classes, 'img_dim': [3, *args.img_size]}

    if args.model_type == 'hyperbolic':
        model = lorentz_model_factory[args.backbone](**model_kwargs)
    else:
        model = model_factory[args.backbone](**model_kwargs)

    trainer = Trainer(model, device, args)

    best_val_acc = 0.0
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.validate(val_loader)
        
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
            })
        
        if args.save_model and val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': args,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
            
            if not args.no_wandb:
                wandb.save(checkpoint_path)
                wandb.run.summary['best_val_acc'] = best_val_acc
                wandb.run.summary['best_epoch'] = epoch + 1

    if not args.no_wandb:
        wandb.finish()
    
    return best_val_acc

def main():
    """
    Main function to run training or hyperparameter search.
    """
    base_args = get_args()

    if base_args.use_optuna:
        def objective(trial):
            trial_args = argparse.Namespace(**vars(base_args))
            
            trial_args.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            trial_args.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

            if not trial_args.no_wandb:
                trial_args.wandb_name = f"trial-{trial.number}-{trial_args.backbone}-{trial_args.model_type}"

            print(f"\nStarting trial {trial.number} with params: {trial.params}")
            
            return run_training(trial_args, trial)

        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=base_args.n_trials)

        print("\nStudy statistics: ")
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
