import os
import torch
import numpy as np
from torch.utils.data import Dataset

class IBeamVoxelDataset(Dataset):
    """ Custom PyTorch Dataset for loading voxelized I-Beam data from NPZ files. """
    def __init__(self, file_list, stats):
        """
        Args:
            file_list (list): List of specific NPZ files to include in this dataset.
            stats (dict): A dictionary containing 'mean_y' and 'std_y' tensors.
        """
        self.file_list = file_list
        self.mean_y = stats['mean_y']
        self.std_y = stats['std_y']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            # Input: Binary mask of the I-beam's geometry [1, D, H, W]
            geometry = torch.from_numpy(data['geometry']).float().unsqueeze(0)
            
            # Target: Displacement field [3, D, H, W] for (dx, dy, dz)
            displacement = torch.from_numpy(data['displacement_field']).float()

            # NORMALIZE THE TARGET DISPLACEMENT FIELD ---
            # Reshape mean and std for broadcasting: [3] -> [3, 1, 1, 1]
            mean = self.mean_y.view(3, 1, 1, 1)
            std = self.std_y.view(3, 1, 1, 1)
            normalized_displacement = (displacement - mean) / std
            
        return geometry, normalized_displacement