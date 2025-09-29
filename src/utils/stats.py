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
            displacement = torch.from_numpy(f['displacement'][:]).to(torch.float32)
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
            displacement = torch.from_numpy(data['displacement_field']).to(torch.float32)
            # Sum over all spatial dimensions, keep the channel dimension
            mean_accumulator += displacement.sum(dim=(1, 2, 3))
            total_voxels += displacement.shape[1] * displacement.shape[2] * displacement.shape[3]
    
    mean_y = mean_accumulator / total_voxels

    # --- Pass 2: Calculate the standard deviation ---
    std_accumulator = torch.zeros(3)
    for npz_path in tqdm(file_list, desc="Voxel Stats (Pass 2/2: Std)"):
        with np.load(npz_path) as data:
            displacement = torch.from_numpy(data['displacement_field']).to(torch.float32)
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

def compute_and_save_param_stats(file_list, stats_path):
    """
    Computes mean/std for the 'simulation_params' saved in NPZ files.
    """
    if os.path.exists(stats_path):
        print(f"Parameter statistics file already exists at {stats_path}. Loading stats.")
        stats = torch.load(stats_path)
        return stats['mean_x_params'], stats['std_x_params']

    print(f"Computing parameter statistics from {len(file_list)} training files...")
    
    all_params = []
    for npz_path in tqdm(file_list, desc="Reading files for param stats"):
        with np.load(npz_path) as data:
            params = torch.from_numpy(data['simulation_params'])
            all_params.append(params)

    all_params = torch.stack(all_params, dim=0)
    mean_x_params = all_params.mean(dim=0)
    std_x_params = all_params.std(dim=0)
    std_x_params[std_x_params == 0] = 1.0

    stats = {'mean_x_params': mean_x_params, 'std_x_params': std_x_params}
    torch.save(stats, stats_path)
    
    print(f"Parameter statistics computed and saved to {stats_path}")
    print(f"  - Mean (x_params): {mean_x_params.numpy()}")
    print(f"  - Std (x_params):  {std_x_params.numpy()}")
    
    return mean_x_params, std_x_params

def compute_and_save_graph_input_stats(file_list, stats_path, use_one_hot=False):
    """
    Computes the mean and standard deviation of the GNN's input node features.
    This function is now flexible and can handle both multi-modal (one-hot) and
    unimodal (scalar) feature constructions.

    Args:
        file_list (list): List of paths to the .h5 files (training set).
        stats_path (str): Path to save the computed statistics (.pt file).
        use_one_hot (bool): Flag to determine the feature construction method.
    """
    if os.path.exists(stats_path):
        print(f"Graph input statistics file already exists at {stats_path}. Loading stats.")
        stats = torch.load(stats_path)
        return stats['mean_x'], stats['std_x']

    print(f"Computing graph input statistics from {len(file_list)} training files...")
    print(f"Feature construction mode: {'One-Hot (Multi-modal)' if use_one_hot else 'Scalar (Unimodal)'}")
    
    all_features = []
    
    for h5_path in tqdm(file_list, desc="Reading files for input stats"):
        with h5py.File(h5_path, 'r') as f:
            node_coords = torch.from_numpy(f['node_coordinates'][:]).to(torch.float32)
            num_nodes = node_coords.shape[0]
            load_type_str = f.attrs.get('load_type', 'bending_y')

            # --- This is the new conditional logic ---
            if use_one_hot:
                # Build 16-dimensional feature vector for the GENERALIST model
                one_hot_map = {"bending_y": [1.0, 0.0, 0.0], "bending_x": [0.0, 1.0, 0.0], "torsion":   [0.0, 0.0, 1.0]}
                one_hot_load_type = torch.tensor(one_hot_map.get(load_type_str, [1.0, 0.0, 0.0]), dtype=torch.float32)
                params_list = [
                    f.attrs.get('beam_length', 300.0), f.attrs.get('flange_width', 100.0),
                    f.attrs.get('flange_thickness', 15.0), f.attrs.get('web_thickness', 10.0),
                    f.attrs.get('beam_depth', 150.0), f.attrs.get('fillet_radius', 12.0),
                    f.attrs.get('youngs_modulus', 2.1e11), f.attrs.get('poissons_ratio', 0.3),
                    f.attrs.get('force_magnitude', 1500.0),
                    {"uniform": 0.0, "linear_y": 1.0}.get(f.attrs.get('load_distribution', 'uniform'), 0.0)
                ]
                params = torch.tensor(params_list, dtype=torch.float32)
                params_per_node = params.unsqueeze(0).repeat(num_nodes, 1)
                one_hot_per_node = one_hot_load_type.unsqueeze(0).repeat(num_nodes, 1)
                node_features = torch.cat([node_coords, params_per_node, one_hot_per_node], dim=1)
            else:
                # Build 14-dimensional feature vector for the SPECIALIST model
                load_type_map = {"bending_y": 0.0, "bending_x": 1.0, "torsion": 2.0}
                params_list = [
                    f.attrs.get('beam_length', 300.0), f.attrs.get('flange_width', 100.0),
                    f.attrs.get('flange_thickness', 15.0), f.attrs.get('web_thickness', 10.0),
                    f.attrs.get('beam_depth', 150.0), f.attrs.get('fillet_radius', 12.0),
                    f.attrs.get('youngs_modulus', 2.1e11), f.attrs.get('poissons_ratio', 0.3),
                    f.attrs.get('force_magnitude', 1500.0),
                    load_type_map.get(load_type_str, 0.0),
                    {"uniform": 0.0, "linear_y": 1.0}.get(f.attrs.get('load_distribution', 'uniform'), 0.0)
                ]
                params = torch.tensor(params_list, dtype=torch.float32)
                params_per_node = params.unsqueeze(0).repeat(num_nodes, 1)
                node_features = torch.cat([node_coords, params_per_node], dim=1)

            all_features.append(node_features)

    all_features = torch.cat(all_features, dim=0)
    mean_x = all_features.mean(dim=0)
    std_x = all_features.std(dim=0)
    std_x[std_x == 0] = 1.0

    stats = {'mean_x': mean_x, 'std_x': std_x}
    torch.save(stats, stats_path)
    
    print(f"Graph input statistics computed and saved to {stats_path}")
    print(f"  - Mean (x): {mean_x.numpy()}")
    print(f"  - Std (x):  {std_x.numpy()}")
    
    return mean_x, std_x