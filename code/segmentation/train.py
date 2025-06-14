# -----------------------------------------------------
# Change working directory to parent HyperbolicCV/code
import os
import sys

working_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")
os.chdir(working_dir)

lib_path = os.path.join(working_dir)
sys.path.append(lib_path)
# -----------------------------------------------------

import torch
from torch.nn import functional as F
from torch.nn import DataParallel

import configargparse
from tqdm import tqdm

import random
import numpy as np

from utils.initialize import (  
    load_checkpoint,
    select_model,
    select_optimizer,
    select_dataset
)
from lib.utils.utils import AverageMeter

from utils.eval import compute_iou

def getArguments():
    """ Parses command-line options. """
    parser = configargparse.ArgumentParser(description='VAE training', add_help=True)

    parser.add_argument('--debug', action='store_true', help="One batch overfitting")
    parser.add_argument('-c', '--config_file', required=False, default=None, is_config_file=True, type=str, 
                        help="Path to config file.")

    # Output settings
    parser.add_argument('--exp_name', default="test", type=str, 
                        help="Name of the experiment.")
    parser.add_argument('--output_dir', default=None, type=str, 
                        help = "Path for output files (relative to working directory).")

    # General settings
    parser.add_argument('--device', default="cuda:0", type=lambda s: [str(item) for item in s.replace(' ','').split(',')],
                        help="List of devices split by comma (e.g. cuda:0,cuda:1), can also be single device or 'cpu')")
    parser.add_argument('--dtype', default='float32', type=str, choices=["float32", "float64"], 
                        help="Set floating point precision.")
    parser.add_argument('--seed', default=1, type=int, 
                        help="Set seed for deterministic training.")
    parser.add_argument('--load_checkpoint', default=None, type=str, 
                        help = "Path to model checkpoint (weights, optimizer, epoch).")

    # General training hyperparameters
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help="Number of training epochs.")
    parser.add_argument('--batch_size', default=100, type=int, 
                        help="Training batch size.")
    parser.add_argument('--lr', default=5e-4, type=float, 
                        help="Training learning rate.")
    parser.add_argument('--weight_decay', default=0, type=float, 
                        help="Weight decay (L2 regularization)")
    parser.add_argument('--optimizer', default="RiemannianAdam", type=str, choices=["RiemannianAdam", "RiemannianSGD", "Adam"], 
                        help="Optimizer for training.")
    parser.add_argument('--use_lr_scheduler', action='store_true',
                        help="If learning rate should be reduced after step epochs using a LR scheduler.")
    parser.add_argument('--lr_scheduler_step', default=50, type=int,
                        help="Step size of LR scheduler.")
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float,
                        help="Gamma parameter of LR scheduler.")

    # General validation/testing hyperparameters
    parser.add_argument('--batch_size_test', default=128, type=int, 
                        help = "Validation/Testing batch size.")
    
    # General VAE hyperparameters (that I'll be using as segmentation hyperparameters)
    parser.add_argument('--model', default='L-Seg', type=str, choices=["L-Seg", "EL-Seg", "EP-Seg", "E-Seg"], 
                        help = "Select a model architecture.")
    parser.add_argument('--enc_layers', default=4, type=int, 
                        help = "Number of convolutional layers in encoder.")
    parser.add_argument('--dec_layers', default=3, type=int, 
                        help = "Number of transposed convolutional layers in decoder.")
    parser.add_argument('--initial_filters', default=64, type=int, 
                        help = "Number of output filters of first convolutional layer. Gets doubled with each conv. layer.")

    # Hyperbolic manifold settings
    parser.add_argument('--learn_K', action='store_true',
                        help="Set a learnable curvature of hyperbolic space for the hyperbolic models.")
    parser.add_argument('--embed_K', default=1.0, type=float, 
                        help = "Initial curvature of embedding space of hybrid HNNs (geoopt.K=-1/K).")

    # HCNN
    parser.add_argument('--enc_K', default=1.0, type=float, 
                        help = "Initial curvature of encoder space (geoopt.K=-1/K).")
    parser.add_argument('--dec_K', default=1.0, type=float, 
                        help = "Initial curvature of decoder space (geoopt.K=-1/K).")
    
    # Dataset settings
    parser.add_argument('--dataset', default='VOCSegmentation', type=str, choices=["VOC"], 
                        help = "Select a dataset.")


    args = parser.parse_args()

    return args


def main(args):
    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    print("Running experiment: " + args.exp_name)

    print("Arguments:")
    print(args)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, out_dim = select_dataset(args)

    print("Creating model...")
    model = select_model(out_dim, args)
    model = model.to(device)
    print('-> Number of model params: {} (trainable: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    print("Creating optimizer...")
    optimizer, lr_scheduler = select_optimizer(model, args)

    start_epoch = 0
    if args.load_checkpoint is not None:
        print("Loading model checkpoint from {}".format(args.load_checkpoint))
        model, optimizer, lr_scheduler, start_epoch = load_checkpoint(model, optimizer, lr_scheduler, args)

    # model = DataParallel(model, device_ids=args.device)

    print("Training...")
    global_step = start_epoch*len(train_loader)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()

        log_loss = AverageMeter("Loss", ":.4e")

        for i, (x,y) in tqdm(enumerate(train_loader)):
            # ------- Start iteration -------
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            y_resized = F.interpolate(y.float(), size=y_hat.shape[2:], mode='nearest').long().squeeze(1)
            loss = F.cross_entropy(y_hat, y_resized, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                log_loss.update(loss.detach().item())

            global_step += 1

            if args.debug and i == 0:
                break
            # ------- End iteration -------

        # ------- Start validation and logging -------
        with torch.no_grad():
            if lr_scheduler is not None:
                lr_scheduler.step()

            metrics_dict = {
                'Loss' : log_loss.avg,
            }

            # Validation
            metrics_dict_val = evaluate(model, val_loader, out_dim, device, enable_debug=args.debug)

            metric_string = ""
            for key in metrics_dict.keys():
                metric_string += ", {} = {:.4f}".format(key, metrics_dict[key])

            for key in metrics_dict_val.keys():
                metric_string += ", Val_{} = {:.4f}".format(key, metrics_dict_val[key])

            print("Epoch {}/{}: {}".format(epoch+1, args.num_epochs, metric_string))
        # ------- End validation and logging -------

    print("-----------------\nTraining finished\n-----------------")
    if (args.output_dir is not None):
        model.cpu()
        save_path = args.output_dir + "/final_" + args.exp_name + ".pth"
        torch.save({
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'epoch': epoch,
            'args': args,
        }, save_path)

        print("Model saved to " + save_path)
    else:
        print("Model not saved.")

    print("Testing...")
    test(model, test_loader, out_dim, device)
    

@torch.no_grad()
def test(model, dataloader_test, out_dim, device):
    """ Tests models performance using Losses and FID """
    metrics_dict = evaluate(model, dataloader_test, out_dim, device)

    metric_string = ""
    for key in metrics_dict.keys():
        metric_string += ", {} = {:.4f}".format(key, metrics_dict[key])

    print("Testing: {}".format(metric_string))

@torch.no_grad()
def evaluate(model, dataloader, num_classes, device, enable_debug):
    """ Evaluates model performance using loss and mIoU """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            y_resized = F.interpolate(y.float(), size=y_hat.shape[2:], mode='nearest').long().squeeze(1)
            loss = F.cross_entropy(y_hat, y_resized, ignore_index=255)
            total_loss += loss.item()

            preds = torch.argmax(y_hat, dim=1)
            total_iou += compute_iou(preds, y_resized, num_classes)

            num_batches += 1
            if enable_debug and i == 0:
                break

    return {
        'Loss': total_loss / num_batches,
        'mIoU%': total_iou / num_batches * 100.0
    }

# ----------------------------------
if __name__ == '__main__':
    args = getArguments()

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
    elif args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        raise "Wrong dtype in configuration -> " + args.dtype

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            print("Create missing output directory...")
            os.mkdir(args.output_dir)

    main(args)

