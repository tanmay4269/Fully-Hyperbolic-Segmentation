import torch
from torchvision import datasets, transforms

from torch.utils.data import DataLoader

from lib.geoopt.optim import RiemannianAdam, RiemannianSGD
from torch.optim.lr_scheduler import StepLR

# from models.LVAE import LVAE
from models.ESeg import ESeg


def load_checkpoint(model, optimizer, lr_scheduler, args):
    """ Loads a checkpoint from file-system. """

    checkpoint = torch.load(args.load_checkpoint, map_location='cpu')

    model.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint:
        if checkpoint['args'].optimizer == args.optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
            for group in optimizer.param_groups:
                group['lr'] = args.lr

            if (lr_scheduler is not None) and ('lr_scheduler' in checkpoint):
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("Warning: Could not load optimizer and lr-scheduler state_dict. Different optimizer in configuration ({}) and checkpoint ({}).".format(args.optimizer, checkpoint['args'].optimizer))

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, lr_scheduler, epoch

def load_model_checkpoint(model, checkpoint_path):
    """ Loads a checkpoint from file-system. """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    return model

def select_model(out_dim, args):
    """ Selects and sets up an available model and returns it. """

    if False and args.debug:
        print("[Debug] Loading smp.Unet instead of your model choice")
        import segmentation_models_pytorch as smp
 
        model = smp.Unet(
            encoder_name="resnet18", 
            # encoder_weights="imagenet",
            in_channels=3,
            classes=21
        )
    elif args.model == "E-Seg":
        model = ESeg(
            args.enc_layers, 
            args.dec_layers, 
            out_dim=out_dim,
            initial_filters=args.initial_filters,
            latent_distr="euclidean"
        )
    else:
        raise "Model not found. Wrong model in configuration... -> " + args.model

    return model

def select_optimizer(model, args):
    """ Selects and sets up an available optimizer and returns it. """

    if args.optimizer == "RiemannianAdam":
        optimizer = RiemannianAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "RiemannianSGD":
        optimizer = RiemannianSGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, stabilize=1)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise "Optimizer not found. Wrong optimizer in configuration... -> " + args.model

    if args.use_lr_scheduler:
        lr_scheduler = StepLR(
            optimizer, step_size=args.lr_scheduler_step, gamma=args.lr_scheduler_gamma
        )
    else:
        lr_scheduler = None

    return optimizer, lr_scheduler


def select_dataset(args):
    """ Selects an available dataset and returns PyTorch dataloaders for training, validation and testing. """
    
    if args.dataset == 'VOC':
        from torchvision.datasets import VOCSegmentation

        img_size = (512, 512)  
        out_dim = 21  # Known about VOC

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

        train_set = VOCSegmentation(
            'data/pascal_voc',
            image_set='train',
            # download=True,
            transform=train_transform,
            target_transform=target_transform,
        )

        full_val = VOCSegmentation(
            'data/pascal_voc',
            image_set='val',
            # download=True,
            transform=val_transform,
            target_transform=target_transform,
        )

        num_val = int(0.5 * len(full_val))
        num_test = len(full_val) - num_val
        val_set, test_set = torch.utils.data.random_split(
            full_val,
            [num_val, num_test], 
            generator=torch.Generator().manual_seed(1)
        )
    else:
        raise "Selected dataset '{}' not available.".format(args.dataset)
    
    # Dataloader
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=(not args.debug),
    )
    val_loader = DataLoader(
        val_set if not args.debug else train_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    )
    test_loader = DataLoader(
        test_set if not args.debug else train_set, 
        batch_size=args.batch_size_test, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False,
    ) 
    
    return train_loader, val_loader, test_loader, out_dim