# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

working_dir = Path(__file__).parent.parent.absolute()
os.chdir(working_dir)
sys.path.append(str(working_dir))
# -----------------------------------------------------

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import configargparse
from tqdm import tqdm
import random
import numpy as np
from datetime import datetime

from utils.initialize import (
    load_checkpoint,
    select_model,
    select_optimizer,
    select_dataset
)
from lib.utils.utils import AverageMeter, compute_iou


class SegmentationTrainer:
    def __init__(self, args: configargparse.Namespace):
        self.args = args
        self.device = args.device[0]
        self.global_step = 0
        self.setup_environment()
        self.setup_logging()
        self.setup_model_and_data()

    def setup_environment(self) -> None:
        """Setup random seeds and CUDA device."""
        torch.manual_seed(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()

    def setup_logging(self) -> None:
        """Initialize TensorBoard and experiment directories."""
        if self.args.output_dir:
            self.output_dir = Path(self.args.output_dir)
            self.output_dir.mkdir(exist_ok=True)
            
            # Create unique experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = self.output_dir / f"{self.args.exp_name}_{timestamp}"
            self.exp_dir.mkdir(exist_ok=True)
            
            # Setup TensorBoard
            self.writer = SummaryWriter(log_dir=str(self.exp_dir / "tensorboard"))
            
            # Save config
            with open(self.exp_dir / "config.txt", "w") as f:
                f.write(str(self.args))
        else:
            self.writer = None

    def setup_model_and_data(self) -> None:
        """Initialize model, datasets, and optimization components."""
        print("Loading dataset...")
        self.train_loader, self.val_loader, self.test_loader, out_dim = select_dataset(self.args)

        print("Creating model...")
        self.model = select_model(out_dim, self.args)
        self.model = self.model.to(self.device)
        self._log_model_info()

        print("Creating optimizer...")
        self.optimizer, self.lr_scheduler = select_optimizer(self.model, self.args)

        if self.args.load_checkpoint:
            self._load_checkpoint()

    def _log_model_info(self) -> None:
        """Log model parameters information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'-> Number of model params: {total_params:,} (trainable: {trainable_params:,})')

    def _load_checkpoint(self) -> None:
        """Load model checkpoint if specified."""
        print(f"Loading model checkpoint from {self.args.load_checkpoint}")
        checkpoint_data = load_checkpoint(self.model, self.optimizer, self.lr_scheduler, self.args)
        self.model, self.optimizer, self.lr_scheduler, self.global_step = checkpoint_data

    def train(self) -> None:
        """Main training loop."""
        print(f"Starting training experiment: {self.args.exp_name}")
        
        for epoch in range(self.global_step, self.args.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader, prefix="Val")
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if self.args.output_dir:
                self._save_checkpoint(epoch)

        print("-----------------\nTraining finished\n-----------------")
        
        # Final test evaluation
        print("Running final test evaluation...")
        test_metrics = self.evaluate(self.test_loader, prefix="Test")
        self._log_metrics(self.args.num_epochs, {}, test_metrics)

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        log_loss = AverageMeter("Loss", ":.4e")
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader),
                   desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for i, (x, y) in pbar:
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y.squeeze(1).long(), ignore_index=255)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Logging
            log_loss.update(loss.item())
            if self.writer:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{log_loss.avg:.4f}'})
            
            self.global_step += 1
            if self.args.debug and i == 0:
                break
                
        return {'Loss': log_loss.avg}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, prefix: str = "") -> Dict[str, float]:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"{prefix} Evaluation")
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            
            loss = F.cross_entropy(y_hat, y.squeeze(1).long(), ignore_index=255)
            total_loss += loss.item()
            
            preds = torch.argmax(y_hat, dim=1)
            total_iou += compute_iou(preds, y.squeeze(1))
            
            num_batches += 1
            if self.args.debug and i == 0:
                break
        
        metrics = {
            'Loss': total_loss / num_batches,
            'mIoU%': total_iou / num_batches * 100.0
        }
        
        return metrics

    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                    eval_metrics: Dict[str, float]) -> None:
        """Log metrics to console and TensorBoard."""
        # Console logging
        metrics_str = f"Epoch {epoch+1}/{self.args.num_epochs}"
        for k, v in train_metrics.items():
            metrics_str += f", {k}: {v:.4f}"
        for k, v in eval_metrics.items():
            metrics_str += f", {k}: {v:.4f}"
        print(metrics_str)
        
        # TensorBoard logging
        if self.writer:
            for k, v in train_metrics.items():
                self.writer.add_scalar(f'train/{k}', v, epoch)
            for k, v in eval_metrics.items():
                self.writer.add_scalar(f'eval/{k}', v, epoch)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'epoch': epoch,
            'args': self.args,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def parse_args() -> configargparse.Namespace:
    """Parse command line arguments."""
    parser = configargparse.ArgumentParser(description='Segmentation Training', add_help=True)
    
    # Debug and config
    parser.add_argument('--debug', action='store_true', help="One batch overfitting")
    parser.add_argument('-c', '--config_file', required=False, default=None, 
                       is_config_file=True, type=str, help="Path to config file.")
    
    # Output settings
    parser.add_argument('--exp_name', default="test", type=str, help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str, help="Path for output files.")
    
    # Training settings
    parser.add_argument('--device', default="cuda:0", 
                       type=lambda s: [str(item) for item in s.replace(' ','').split(',')],
                       help="List of devices (e.g. cuda:0,cuda:1)")
    parser.add_argument('--dtype', default='float32', type=str, 
                       choices=["float32", "float64"], help="Floating point precision.")
    parser.add_argument('--seed', default=1, type=int, help="Random seed.")
    parser.add_argument('--load_checkpoint', default=None, type=str, help="Path to checkpoint.")
    
    # Training hyperparameters
    parser.add_argument('--num_epochs', default=100, type=int, help="Number of epochs.")
    parser.add_argument('--batch_size', default=100, type=int, help="Training batch size.")
    parser.add_argument('--lr', default=5e-4, type=float, help="Learning rate.")
    parser.add_argument('--weight_decay', default=0, type=float, help="Weight decay")
    parser.add_argument('--optimizer', default="RiemannianAdam", type=str,
                       choices=["RiemannianAdam", "RiemannianSGD", "Adam"])
    parser.add_argument('--use_lr_scheduler', action='store_true', help="Use LR scheduler")
    parser.add_argument('--lr_scheduler_step', default=50, type=int, help="LR scheduler step")
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float, help="LR scheduler gamma")
    
    # Validation/Testing
    parser.add_argument('--batch_size_test', default=128, type=int, help="Test batch size.")
    
    # Model architecture
    parser.add_argument('--model', default='L-Seg', type=str,
                       choices=["L-Seg", "EL-Seg", "EP-Seg", "E-Seg"])
    parser.add_argument('--enc_layers', default=4, type=int, help="Encoder layers")
    parser.add_argument('--dec_layers', default=3, type=int, help="Decoder layers")
    parser.add_argument('--initial_filters', default=64, type=int, help="Initial filters")
    
    # Hyperbolic settings
    parser.add_argument('--learn_K', action='store_true', help="Learn curvature")
    parser.add_argument('--embed_K', default=1.0, type=float, help="Initial embedding curvature")
    parser.add_argument('--enc_K', default=1.0, type=float, help="Initial encoder curvature")
    parser.add_argument('--dec_K', default=1.0, type=float, help="Initial decoder curvature")
    
    # Dataset
    parser.add_argument('--dataset', default='VOCSegmentation', type=str, choices=["VOC"])
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set precision
    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    
    # Initialize trainer
    trainer = SegmentationTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

