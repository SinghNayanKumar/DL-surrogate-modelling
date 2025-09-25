import os
import torch
import h5py
from tqdm import tqdm
import numpy as np

def compute_and_save_stats(file_list, stats_path):
    """
    Computes the mean and standard deviation of the 'displacement' dataset
    across a list of H5 files and saves them to a file.

    Args:
        file_list (list): List of paths to the .h5 files (training set).
        stats_path (str): Path to save the computed statistics (.pt file).
    """
    if os.path.exists(stats_path):
        print(f"Statistics file already exists at {stats_path}. Loading stats.")
        stats = torch.load(stats_path)
        return stats['mean_y'], stats['std_y']

    print(f"Computing dataset statistics from {len(file_list)} training files...")
    
    # Accumulate all displacement tensors in a list
    all_displacements = []
    for h5_path in tqdm(file_list, desc="Reading files for stats"):
        with h5py.File(h5_path, 'r') as f:
            displacement = torch.from_numpy(f['displacement'][:])
            all_displacements.append(displacement)

    # Concatenate all tensors into one large tensor
    # This might be memory-intensive for very large datasets
    all_displacements = torch.cat(all_displacements, dim=0)

    # Compute mean and std over all nodes (dim=0)
    # The result will be a tensor of shape [3] for (dx, dy, dz)
    mean_y = all_displacements.mean(dim=0)
    std_y = all_displacements.std(dim=0)
    
    # Add a small epsilon to std to prevent division by zero if a feature is constant
    std_y[std_y == 0] = 1.0

    stats = {'mean_y': mean_y, 'std_y': std_y}
    torch.save(stats, stats_path)
    
    print(f"Statistics computed and saved to {stats_path}")
    print(f"  - Mean (y): {mean_y.numpy()}")
    print(f"  - Std (y):  {std_y.numpy()}")
    
    return mean_y, std_y

def compute_and_save_voxel_stats(file_list, stats_path):
    """
    Computes the mean and standard deviation of the 'displacement_field'
    across a list of NPZ files and saves them to a file.
    This uses a memory-efficient two-pass approach.
    """
    if os.path.exists(stats_path):
        print(f"Voxel statistics file already exists at {stats_path}. Loading stats.")
        stats = torch.load(stats_path)
        return stats['mean_y'], stats['std_y']

    print(f"Computing voxel dataset statistics from {len(file_list)} training files...")
    
    # --- Pass 1: Calculate the mean ---
    mean_accumulator = torch.zeros(3) # For dx, dy, dz
    total_voxels = 0
    for npz_path in tqdm(file_list, desc="Voxel Stats (Pass 1/2: Mean)"):
        with np.load(npz_path) as data:
            # Shape: (3, D, H, W)
            displacement = torch.from_numpy(data['displacement_field'])
            # Sum over all spatial dimensions, keep the channel dimension
            mean_accumulator += displacement.sum(dim=(1, 2, 3))
            total_voxels += displacement.shape[1] * displacement.shape[2] * displacement.shape[3]
    
    mean_y = mean_accumulator / total_voxels

    # --- Pass 2: Calculate the standard deviation ---
    std_accumulator = torch.zeros(3)
    for npz_path in tqdm(file_list, desc="Voxel Stats (Pass 2/2: Std)"):
        with np.load(npz_path) as data:
            displacement = torch.from_numpy(data['displacement_field'])
            # Reshape mean for broadcasting: [3] -> [3, 1, 1, 1]
            reshaped_mean = mean_y.view(3, 1, 1, 1)
            # Sum of squared differences
            std_accumulator += ((displacement - reshaped_mean) ** 2).sum(dim=(1, 2, 3))
            
    std_y = torch.sqrt(std_accumulator / total_voxels)
    std_y[std_y == 0] = 1.0 # Prevent division by zero

    stats = {'mean_y': mean_y, 'std_y': std_y}
    torch.save(stats, stats_path)
    
    print(f"Voxel statistics computed and saved to {stats_path}")
    print(f"  - Mean (y): {mean_y.numpy()}")
    print(f"  - Std (y):  {std_y.numpy()}")
    
    return mean_y, std_y