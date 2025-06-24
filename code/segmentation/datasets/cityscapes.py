import torch
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision import transforms

class CityscapesDataset:
    """
    Wrapper for the Cityscapes dataset with albumentations support.
    """
    def __init__(self, root, split='train', albumentation_transform=None):
        """
        Initializes the Cityscapes dataset.

        Args:
            root (str): The root directory of the Cityscapes dataset.
            split (str, optional): The dataset split, 'train', 'val', or 'test'. Defaults to 'train'.
            albumentation_transform (callable, optional): Albumentations transform to be applied to images and masks. Defaults to None.
        """
        self.dataset = Cityscapes(root=root, split=split, mode='fine', target_type='semantic')
        self.albumentation_transform = albumentation_transform
        
        self.id_to_train_id = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the transformed image and mask.
        """
        img, mask = self.dataset[idx]
        
        img_np = np.array(img)
        mask_np = np.array(mask)

        # Map the original class IDs to training IDs
        mask_train_id = np.full_like(mask_np, 255, dtype=np.uint8)
        for k, v in self.id_to_train_id.items():
            mask_train_id[mask_np == k] = v
        
        if self.albumentation_transform is None:
            # Default transform if no albumentations transform is provided
            return transforms.ToTensor()(img), torch.from_numpy(mask_train_id).long()

        transformed = self.albumentation_transform(image=img_np, mask=mask_train_id)
        img_transformed = transformed['image']
        mask_transformed = transformed['mask']
        
        if isinstance(mask_transformed, torch.Tensor):
            return img_transformed, mask_transformed.long()
        else:
            return img_transformed, torch.from_numpy(mask_transformed).long()



