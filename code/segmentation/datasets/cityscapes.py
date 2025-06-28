import torch
import numpy as np
from torchvision.datasets import Cityscapes
from torchvision import transforms
import logging
import math
import random
from tqdm import tqdm
import os
import json
import hashlib

class CityscapesDataset:
    """
    Wrapper for the Cityscapes dataset with albumentations support, subsampling, and class balancing.
    """
    def __init__(self, root, split='train', albumentation_transform=None, subsample_percentage=100.0):
        """
        Initializes the Cityscapes dataset.

        Args:
            root (str): The root directory of the Cityscapes dataset.
            split (str, optional): The dataset split, 'train', 'val', or 'test'. Defaults to 'train'.
            albumentation_transform (callable, optional): Albumentations transform to be applied to images and masks. Defaults to None.
            subsample_percentage (float, optional): Percentage of dataset to use. Defaults to 100.0.
        """
        self.dataset = Cityscapes(root=root, split=split, mode='fine', target_type='semantic')
        self.albumentation_transform = albumentation_transform
        self.root = root
        self.split = split
        self.subsample_percentage = subsample_percentage
        
        self.num_classes = 19
        self.id_to_train_id = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
            23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18
        }

        self.indices = list(range(len(self.dataset)))
        self.class_weights = None

        self._analyze_and_subsample(subsample_percentage)

    def _get_cache_path(self):
        """Generate a unique cache path based on dataset parameters."""
        # Create a hash of the relevant parameters
        params = f"{self.root}_{self.split}_{self.subsample_percentage}_{self.num_classes}"
        param_hash = hashlib.md5(params.encode()).hexdigest()
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        return os.path.join(cache_dir, f"cityscapes_analysis_{self.split}_{self.subsample_percentage}_{param_hash}.json")

    def _analyze_and_subsample(self, subsample_percentage):
        """
        Analyzes the dataset for class distribution, performs pixel-level subsampling if requested,
        and calculates class weights based on the subsampled dataset.
        Uses caching to avoid redundant calculations.
        """
        cache_path = self._get_cache_path()
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            try:
                logging.info(f"Loading cached analysis for {self.split} set from {cache_path}")
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                self.indices = cache_data['indices']
                if 'class_weights' in cache_data and cache_data['class_weights'] is not None:
                    self.class_weights = torch.tensor(cache_data['class_weights'])
                
                logging.info(f"Successfully loaded cached analysis. Dataset size: {len(self.indices)} images.")
                return
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}. Performing analysis from scratch.")
        
        logging.info(f"Analyzing {self.split} dataset for pixel distribution...")
        image_pixel_counts = []
        total_pixel_counts = torch.zeros(self.num_classes, dtype=torch.float64)

        for i in tqdm(range(len(self.dataset)), desc=f"Analyzing {self.split} data"):
            _, mask = self.dataset[i]
            mask_np = np.array(mask)
            mask_train_id = np.full_like(mask_np, 255, dtype=np.uint8)
            for k, v in self.id_to_train_id.items():
                mask_train_id[mask_np == k] = v
            
            mask_tensor = torch.from_numpy(mask_train_id)
            
            counts = torch.zeros(self.num_classes, dtype=torch.int64)
            for c in range(self.num_classes):
                counts[c] = (mask_tensor == c).sum()
            
            image_pixel_counts.append(counts)
            total_pixel_counts += counts

        selected_indices = list(range(len(self.dataset)))
        
        if subsample_percentage < 100.0:
            logging.info(f"Performing pixel-level subsampling for {self.split} set to >={subsample_percentage}%...")
            
            target_pixel_counts = total_pixel_counts * (subsample_percentage / 100.0)
            target_pixel_counts[total_pixel_counts == 0] = 1
            
            current_pixel_counts = torch.zeros(self.num_classes, dtype=torch.float64)
            unselected_indices = list(range(len(self.dataset)))
            selected_indices = []

            pbar = tqdm(total=self.num_classes, desc="Greedy subsampling")
            needs_met = current_pixel_counts >= target_pixel_counts
            pbar.update(needs_met.sum().item())
            
            while not torch.all(needs_met):
                best_image_idx = -1
                best_score = -1
                
                needs_vector = torch.max(torch.zeros_like(target_pixel_counts), target_pixel_counts - current_pixel_counts)
                
                for i in unselected_indices:
                    useful_pixels = torch.min(image_pixel_counts[i].double(), needs_vector)
                    score = (useful_pixels / (total_pixel_counts + 1e-9)).sum()

                    if score > best_score:
                        best_score = score
                        best_image_idx = i
                
                if best_image_idx == -1 or best_score <= 0:
                    logging.warning("Greedy subsampling stopped before all targets met.")
                    break
                
                selected_indices.append(best_image_idx)
                unselected_indices.remove(best_image_idx)
                
                current_pixel_counts += image_pixel_counts[best_image_idx].double()
                
                newly_met = (current_pixel_counts >= target_pixel_counts) & (~needs_met)
                pbar.update(newly_met.sum().item())
                needs_met = current_pixel_counts >= target_pixel_counts

            pbar.close()
            self.indices = sorted(selected_indices)
            
            logging.info(f"Subsampling for {self.split} set complete.")
            logging.info(f"Total images before: {len(self.dataset)}, after: {len(self.indices)}")
            header = f"{'Class':<10} | {'Target %':<10} | {'Actual %':<10} | {'Original Pixels':<18} | {'Subsampled Pixels':<18}"
            logging.info(header)
            logging.info("-" * len(header))
            
            final_subsampled_counts = torch.zeros(self.num_classes, dtype=torch.float64)
            for i in self.indices:
                final_subsampled_counts += image_pixel_counts[i].double()

            for c in range(self.num_classes):
                original_c = total_pixel_counts[c].item()
                subsampled_c = final_subsampled_counts[c].item()
                
                if original_c > 0:
                    percentage = (subsampled_c / original_c) * 100
                    logging.info(f"{c:<10} | {subsample_percentage:9.2f}% | {percentage:9.2f}% | {int(original_c):<18} | {int(subsampled_c):<18}")
                    if percentage < subsample_percentage:
                        logging.warning(f"Class {c} has {percentage:.2f}% of pixels, less than target {subsample_percentage}%.")
                else:
                    logging.info(f"{c:<10} | {'N/A':<9} | {'N/A':<9} | {0:<18} | {0:<18}")
        else:
            self.indices = selected_indices
        
        # Calculate class weights based on the subsampled dataset (or full dataset if no subsampling)
        if self.split == 'train':
            logging.info("Calculating class weights based on the selected dataset...")
            
            subsampled_pixel_counts = torch.zeros(self.num_classes, dtype=torch.float64)
            for i in self.indices:
                subsampled_pixel_counts += image_pixel_counts[i].double()
                
            total_pixels = subsampled_pixel_counts.sum()
            if total_pixels > 0:
                class_freq = subsampled_pixel_counts / total_pixels
                self.class_weights = 1.0 / torch.log(1.02 + class_freq)
                logging.info(f"Class weights calculated: {self.class_weights}")
            else:
                logging.warning("No labeled pixels found to calculate class weights.")
        
        # Save to cache
        try:
            cache_data = {
                'indices': self.indices,
                'class_weights': self.class_weights.tolist() if self.class_weights is not None else None
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            logging.info(f"Analysis results cached to {cache_path}")
        except Exception as e:
            logging.warning(f"Failed to cache analysis results: {e}")

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple: A tuple containing the transformed image and mask.
        """
        original_idx = self.indices[idx]
        img, mask = self.dataset[original_idx]
        
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



