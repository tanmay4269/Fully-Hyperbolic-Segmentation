import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

class VOCDataset:
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