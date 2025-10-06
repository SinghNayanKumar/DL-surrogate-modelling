import torch
import glob
import time
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from src.datasets.graph_dataset import IBeamGraphDataset
from src.datasets.voxel_dataset import IBeamVoxelDataset
from src.models.gnn_variants import GCN_Surrogate, GAT_Surrogate, MPNN_Surrogate
from src.models.graph_transformer import GraphTransformer_Surrogate
from src.models.unet_variants import UNet3D, UNet3D_Small

MODEL_MAPPING = {
    'gcn': GCN_Surrogate, 'gat': GAT_Surrogate, 'mpnn': MPNN_Surrogate,
    'transformer': GraphTransformer_Surrogate,
    'unet': UNet3D, 'unet_small': UNet3D_Small,
}

def evaluate_single_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type = config['model_type']
    is_gnn = model_type in ['gcn', 'gat', 'mpnn', 'transformer']
    data_ext = '*.h5' if is_gnn else '*.npz'

    stats_y = torch.load(config['stats_y_path'])
    stats_x = torch.load(config['stats_x_path'])
    stats = {
        'mean_y': stats_y.get('mean_y'), 'std_y': stats_y.get('std_y'),
        'mean_x': stats_x.get('mean_x'), 'std_x': stats_x.get('std_x'),
        'mean_x_params': stats_x.get('mean_x_params'), 'std_x_params': stats_x.get('std_x_params'),
    }
    mean_y, std_y = stats['mean_y'].to(device), stats['std_y'].to(device)

    all_files = sorted(glob.glob(f"{config['data_dir']}/{data_ext}"))
    if not all_files:
        raise FileNotFoundError(f"No files found at {config['data_dir']}/{data_ext}")
        
    _, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    save_for_viz = config.get('save_for_visualization', False)
    viz_indices = config.get('visualization_indices', [0])
    viz_output_dir = config.get('visualization_output_dir', 'results/visualizations')
    if save_for_viz:
        os.makedirs(viz_output_dir, exist_ok=True)

    if is_gnn:
        test_dataset = IBeamGraphDataset(root_dir=config['data_dir'], h5_file_list=test_files, stats=stats, use_one_hot=config.get('ablation_one_hot', False))
        test_loader = PyGDataLoader(test_dataset, batch_size=1)
    else:
        test_dataset = IBeamVoxelDataset(file_list=test_files, stats=stats)
        test_loader = TorchDataLoader(test_dataset, batch_size=1)

    model_class = MODEL_MAPPING[model_type]
    
    if is_gnn:
        num_node_features = 16 if config.get('ablation_one_hot') else 14
        model = model_class(node_in_features=num_node_features, node_out_features=3, **config).to(device)
    else:
        num_input_channels = 1 + 11 
        model = model_class(in_channels=num_input_channels, out_channels=3, **config).to(device)
    
    model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
    model.eval()

    all_preds, all_targets = [], []
    total_inference_time = 0.0
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), desc="Evaluating", total=len(test_loader)):
            start_time = time.perf_counter()
            if is_gnn:
                data = data.to(device)
                preds = model(data)
                targets = data.y
            else:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
            end_time = time.perf_counter()
            
            if is_gnn:
                unnorm_preds = preds * std_y + mean_y
                unnorm_targets = targets * std_y + mean_y
            else:
                unnorm_preds = preds * std_y.view(1, 3, 1, 1, 1) + mean_y.view(1, 3, 1, 1, 1)
                unnorm_targets = targets * std_y.view(1, 3, 1, 1, 1) + mean_y.view(1, 3, 1, 1, 1)

            if save_for_viz and i in viz_indices:
                model_name_safe = config['model_name'].replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
                output_path = os.path.join(viz_output_dir, f"{model_name_safe}_testsample_{i}.npz")
                
                save_payload = {
                    'prediction': unnorm_preds.cpu().numpy().squeeze(),
                    'ground_truth': unnorm_targets.cpu().numpy().squeeze()
                }
                
                if is_gnn:
                    save_payload['node_coordinates'] = data.pos.cpu().numpy()
                    save_payload['original_h5_path'] = test_files[i]
                else:
                    # --- THE DEFINITIVE FIX ---
                    # The original bug was saving the processed mask from the 'inputs' tensor.
                    # This code loads the original, unmodified mask directly from the data file.
                    current_npz_path = test_files[i]
                    with np.load(current_npz_path) as original_data:
                        mask_key = 'geometry' if 'geometry' in original_data else 'geometry_mask'
                        if mask_key not in original_data:
                            raise KeyError(f"Fatal: Could not find geometry mask in {current_npz_path}")
                        save_payload['geometry_mask'] = original_data[mask_key]
                    # --- END FIX ---

                    # Infer the path to the original .h5 file for topology
                    base_filename = os.path.basename(current_npz_path).replace('.npz', '.h5')
                    h5_dir = os.path.join(os.path.dirname(config['data_dir']), 'h5_raw_unimodal')
                    inferred_h5_path = os.path.join(h5_dir, base_filename)
                    save_payload['original_h5_path'] = inferred_h5_path

                print(f"\nSaving CORRECTED visualization data for sample {i} to {output_path}")
                np.savez_compressed(output_path, **save_payload)

            all_preds.append(unnorm_preds.cpu().numpy())
            all_targets.append(unnorm_targets.cpu().numpy())
            total_inference_time += (end_time - start_time)

    all_preds = np.concatenate([p.reshape(p.shape[0], -1) for p in all_preds], axis=1) if not is_gnn else np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate([t.reshape(t.shape[0], -1) for t in all_targets], axis=1) if not is_gnn else np.concatenate(all_targets, axis=0)
    all_preds = all_preds.reshape(-1, 3)
    all_targets = all_targets.reshape(-1, 3)

    mae_mm = np.mean(np.abs(all_preds - all_targets)) * 1000
    rl2_percent = 100 * np.linalg.norm(all_preds - all_targets) / np.linalg.norm(all_targets)
    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets, axis=0)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    inference_ms = (total_inference_time / len(test_dataset)) * 1000
    params_M = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    return {'mae_mm': mae_mm, 'rl2_percent': rl2_percent, 'r2_score': r2_score, 'inference_ms': inference_ms, 'params_M': params_M}