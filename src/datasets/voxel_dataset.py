import os
import torch
import numpy as np
from torch.utils.data import Dataset

class IBeamVoxelDataset(Dataset):
    """
    Custom PyTorch Dataset for loading voxelized I-Beam data from NPZ files.
    This version creates a multi-channel input for the U-Net:
     - Channel 0: The binary geometry mask of the I-Beam.
     - Channel 1..N: Constant value channels representing the simulation's physical parameters.
    """
    def __init__(self, file_list, stats):
        """
        Args:
            file_list (list): List of specific NPZ files to include in this dataset.
            stats (dict): A dictionary containing normalization stats for both inputs and outputs.
                          Expects: 'mean_y', 'std_y', 'mean_x_params', 'std_x_params'.
        """
        self.file_list = file_list
        # Normalization stats for the output displacement field
        self.mean_y = stats['mean_y']
        self.std_y = stats['std_y']
        # ### --- CHANGE --- ###
        # Normalization stats for the input simulation parameters
        self.mean_x_params = stats['mean_x_params']
        self.std_x_params = stats['std_x_params']

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            # --- 1. Prepare Model Inputs ---
            
            # Base input: Binary mask of the I-beam's geometry [1, D, H, W]
            geometry = torch.from_numpy(data['geometry']).float().unsqueeze(0)
            
            # ### --- CHANGE (CRITICAL) --- ###
            # Load the simulation parameters you saved in the pre-processing step
            # Shape: [num_params] e.g., [force_mag, youngs_mod]
            params = torch.from_numpy(data['simulation_params']).float()
            
            # Normalize the input parameters (important for stable training)
            normalized_params = (params - self.mean_x_params) / self.std_x_params
            
            # Create a list of all input channels
            input_channels = [geometry]
            
            # For each parameter, create a full 3D channel with its constant value
            # This "informs" every voxel about the global simulation conditions.
            for i in range(normalized_params.shape[0]):
                param_channel = torch.full_like(
                    geometry, # Use geometry as a template for the shape
                    fill_value=normalized_params[i].item()
                )
                input_channels.append(param_channel)
                
            # Concatenate all channels along the channel dimension (dim=0)
            # The final input tensor will have shape [1 + num_params, D, H, W]
            model_input = torch.cat(input_channels, dim=0)

            # --- 2. Prepare Model Target ---
            
            # Target: Displacement field [3, D, H, W] for (dx, dy, dz)
            displacement = torch.from_numpy(data['displacement_field']).float()

            # Normalize the target displacement field
            mean = self.mean_y.view(3, 1, 1, 1)
            std = self.std_y.view(3, 1, 1, 1)
            normalized_displacement = (displacement - mean) / std
            
        return model_input, normalized_displacement