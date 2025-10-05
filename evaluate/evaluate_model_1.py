import os
import time
import torch
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import glob

# --- Import your project's modules ---
from src.datasets.graph_dataset import IBeamGraphDataset
from src.datasets.voxel_dataset import IBeamVoxelDataset
from src.models.gnn_variants import GCN_Surrogate, GAT_Surrogate, MPNN_Surrogate
from src.models.graph_transformer import GraphTransformer_Surrogate
from src.models.unet_variants import UNet3D, UNet3D_Small

MODEL_MAPPING = {
    'gcn': GCN_Surrogate, 'gat': GAT_Surrogate, 'mpnn': MPNN_Surrogate,
    'transformer': GraphTransformer_Surrogate, 'unet': UNet3D, 'unet_small': UNet3D_Small,
}

# --- Helper functions ---
def calculate_relative_l2_error(y_true, y_pred):
    y_true_flat = y_true.reshape(y_true.shape[0], -1)
    y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
    numerator = torch.norm(y_true_flat - y_pred_flat, p=2, dim=1)
    denominator = torch.norm(y_true_flat, p=2, dim=1)
    denominator[denominator < 1e-9] = 1.0
    return (numerator / denominator).mean().item()

def generate_visualizations(samples, output_dir, model_name):
    """
    Generates and saves 3D comparison plots and an error histogram.
    
    Args:
        samples (list of dicts): A list containing evaluation data for a few samples.
                                 Each dict should have 'coords', 'true_disp', 'pred_disp'.
        output_dir (str): Directory to save the plots.
        model_name (str): Name of the model for titles and filenames.
    """
    # 1. Generate 3D comparison plots for a few samples
    for i, sample in enumerate(samples[:3]): # Plot first 3 samples
        coords = sample['coords']
        true_disp = sample['true_disp']
        pred_disp = sample['pred_disp']
        error = np.linalg.norm(true_disp - pred_disp, axis=1)
        
        # Create PyVista point clouds
        cloud_true = pv.PolyData(coords + true_disp)
        cloud_true['Displacement'] = np.linalg.norm(true_disp, axis=1)

        cloud_pred = pv.PolyData(coords + pred_disp)
        cloud_pred['Displacement'] = np.linalg.norm(pred_disp, axis=1)

        cloud_error = pv.PolyData(coords)
        cloud_error['Absolute Error (mm)'] = error
        
        # Setup plotter
        plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[1800, 600])
        plotter.subplot(0, 0)
        plotter.add_mesh(cloud_true, scalars='Displacement', cmap='viridis', scalar_bar_args={'title': 'Disp. (mm)'})
        plotter.add_text("Ground Truth", font_size=12)
        plotter.view_isometric()

        plotter.subplot(0, 1)
        plotter.add_mesh(cloud_pred, scalars='Displacement', cmap='viridis', scalar_bar_args={'title': 'Disp. (mm)'})
        plotter.add_text("Model Prediction", font_size=12)
        plotter.view_isometric()

        plotter.subplot(0, 2)
        plotter.add_mesh(cloud_error, scalars='Absolute Error (mm)', cmap='Reds', scalar_bar_args={'title': 'Error (mm)'})
        plotter.add_text("Absolute Error", font_size=12)
        plotter.view_isometric()

        plotter.link_views() # Link camera controls
        plot_filename = os.path.join(output_dir, f"{model_name}_comparison_sample_{i}.png")
        plotter.screenshot(plot_filename)
        print(f"Saved comparison plot to {plot_filename}")

    # 2. Generate error distribution histogram
    all_errors = np.concatenate([s['error_dist'] for s in samples])
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=50, color='crimson', alpha=0.7)
    plt.title(f'Per-Node Prediction Error Distribution ({model_name})')
    plt.xlabel('Error (True - Predicted Displacement) in mm')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    hist_filename = os.path.join(output_dir, f"{model_name}_error_histogram.png")
    plt.savefig(hist_filename)
    print(f"Saved error histogram to {hist_filename}")
    plt.close()
    

# --- Core Evaluation Function ---
def evaluate_single_model(eval_config):
    """
    Loads a single model and evaluates it on its corresponding test set.

    Args:
        eval_config (dict): A dictionary containing all parameters for this run.

    Returns:
        dict: A dictionary containing the aggregated evaluation metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gnn = eval_config['model_type'] in ['gcn', 'gat', 'mpnn', 'transformer']

    # --- Load Data and Normalization Stats ---
    all_files = sorted(glob.glob(os.path.join(eval_config['data_dir'], '*.h5' if is_gnn else '*.npz')))
    _, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

    stats = {}
    if is_gnn:
        stats_y = torch.load(eval_config['stats_y_path'])
        stats_x = torch.load(eval_config['stats_x_path'])
        stats = {**stats_y, **stats_x}
        mean_y, std_y = stats['mean_y'], stats['std_y']
    else: # U-Net
        stats_y = torch.load(eval_config['stats_y_path'])
        stats_x = torch.load(eval_config['stats_x_path'])
        stats = {'mean_y': stats_y['mean_y'], 'std_y': stats_y['std_y'],
                 'mean_x_params': stats_x['mean_x_params'], 'std_x_params': stats_x['std_x_params']}
        mean_y = stats['mean_y'].view(1, 3, 1, 1, 1).to(device)
        std_y = stats['std_y'].view(1, 3, 1, 1, 1).to(device)

    # --- Create Dataset and DataLoader ---
    if is_gnn:
        test_dataset = IBeamGraphDataset(root_dir=eval_config['data_dir'], h5_file_list=test_files, stats=stats, use_one_hot=eval_config.get('ablation_one_hot', False))
        test_loader = PyGDataLoader(test_dataset, batch_size=eval_config['batch_size'], shuffle=False)
    else:
        test_dataset = IBeamVoxelDataset(file_list=test_files, stats=stats)
        test_loader = TorchDataLoader(test_dataset, batch_size=eval_config['batch_size'], shuffle=False)

    # --- Load Model ---
    model_class = MODEL_MAPPING[eval_config['model_type']]
    if is_gnn:
        num_node_features = 16 if eval_config.get('ablation_one_hot', False) else 14
        model = model_class(node_in_features=num_node_features, node_out_features=3).to(device)
    else:
        num_input_channels = 1 + 11
        model = model_class(in_channels=num_input_channels, out_channels=3, use_attention=eval_config.get('use_attention', False)).to(device)
    
    model.load_state_dict(torch.load(eval_config['checkpoint_path'], map_location=device))
    model.eval()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- Evaluation Loop ---
    all_maes, all_rl2, all_inference_times = [], [], []
    all_true_flat, all_pred_flat = [], []

    with torch.no_grad():
        for data in test_loader:
            if is_gnn:
                data = data.to(device)
                inputs, targets_norm = data, data.y
            else:
                inputs, targets_norm = data
                inputs, targets_norm = inputs.to(device), targets_norm.to(device)
            
            start_time = time.time()
            outputs_norm = model(inputs)
            inference_time = (time.time() - start_time) / targets_norm.shape[0]

            if is_gnn:
                targets = targets_norm * std_y.to(device) + mean_y.to(device)
                outputs = outputs_norm * std_y.to(device) + mean_y.to(device)
            else:
                targets = targets_norm * std_y + mean_y
                outputs = outputs_norm * std_y + mean_y
            
            all_maes.append(mean_absolute_error(targets.flatten().cpu(), outputs.flatten().cpu()))
            all_rl2.append(calculate_relative_l2_error(targets, outputs))
            all_inference_times.append(inference_time)
            all_true_flat.append(targets.flatten().cpu())
            all_pred_flat.append(outputs.flatten().cpu())

    # --- Aggregate and Return Final Results ---
    final_r2 = r2_score(torch.cat(all_true_flat), torch.cat(all_pred_flat))
    
    results = {
        'mae_mm': np.mean(all_maes),
        'rl2_percent': np.mean(all_rl2) * 100,
        'r2_score': final_r2,
        'inference_ms': np.mean(all_inference_times) * 1000,
        'params_M': num_params / 1_000_000
    }
    return results