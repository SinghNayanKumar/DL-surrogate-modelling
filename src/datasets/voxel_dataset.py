import os
import torch
import numpy as np
from torch.utils.data import Dataset

class IBeamVoxelDataset(Dataset):
    """ Custom PyTorch Dataset for loading voxelized I-Beam data from NPZ files. """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            # Input: Binary mask of the I-beam's geometry [1, D, H, W]
            geometry = torch.from_numpy(data['geometry']).float().unsqueeze(0)
            
            # Target: Displacement field [3, D, H, W] for (dx, dy, dz)
            displacement = torch.from_numpy(data['displacement_field']).float()
            
        return geometry, displacement