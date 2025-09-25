import os
import glob
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

# Import all our custom modules
from src.datasets.graph_dataset import IBeamGraphDataset
from src.datasets.voxel_dataset import IBeamVoxelDataset
from src.models.gnn_variants import GCN_Surrogate, GAT_Surrogate, MPNN_Surrogate
from src.models.graph_transformer import GraphTransformer_Surrogate
from src.models.unet_variants import UNet3D
from src.training.trainer import ModelTrainer
from src.utils.stats import compute_and_save_stats, compute_and_save_voxel_stats 

MODEL_MAPPING = {
    'gcn': GCN_Surrogate, 'gat': GAT_Surrogate, 'mpnn': MPNN_Surrogate,
    'transformer': GraphTransformer_Surrogate, 'unet': UNet3D,
}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    is_gnn = args.model_type in ['gcn', 'gat', 'mpnn', 'transformer']
    data_subdir = 'h5_raw' if is_gnn else 'voxelized'
    data_ext = '*.h5' if is_gnn else '*.npz'
    data_dir = os.path.join(args.base_data_dir, data_subdir)
    
    all_files = sorted(glob.glob(os.path.join(data_dir, data_ext)))
    
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1


    # --- COMPUTE AND LOAD NORMALIZATION STATS ---
    stats = {}
    if is_gnn:
        stats_path = os.path.join(args.base_data_dir, 'gnn_stats.pt')
        mean_y, std_y = compute_and_save_stats(train_files, stats_path)
        stats = {'mean_y': mean_y, 'std_y': std_y}
    else: # unet
        stats_path = os.path.join(args.base_data_dir, 'voxel_stats.pt')
        mean_y, std_y = compute_and_save_voxel_stats(train_files, stats_path)
        stats = {'mean_y': mean_y, 'std_y': std_y}

    if is_gnn:
        train_dataset = IBeamGraphDataset(root_dir=data_dir, h5_file_list=train_files, stats=stats)
        val_dataset = IBeamGraphDataset(root_dir=data_dir, h5_file_list=val_files, stats=stats)
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    else: # unet
        train_dataset = IBeamVoxelDataset(file_list=train_files, stats=stats)
        val_dataset = IBeamVoxelDataset(file_list=val_files, stats=stats)
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_files)} test samples.")

    # --- Model Selection ---
    model_class = MODEL_MAPPING[args.model_type]
    if is_gnn:
        model = model_class(node_in_features=3, node_out_features=3, hidden_size=128).to(device)
    else:
        model = model_class(in_channels=1, out_channels=3).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Trainer Configuration ---
    trainer_config = {
        'model_type': args.model_type,
        'experiment_name': f"{args.model_type}_{'pinn' if args.use_pinn else 'data'}",
        'use_pinn': args.use_pinn,
        'pinn_weight': args.pinn_weight
    }
    
    trainer = ModelTrainer(model, train_loader, val_loader, optimizer, device, trainer_config)
    trainer.train(args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FEA Surrogate Model Training")
    parser.add_argument('--model_type', type=str, required=True, choices=MODEL_MAPPING.keys())
    parser.add_argument('--data_dir', type=str, default='data/h5_raw', help="Directory for data files")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    # PINN specific arguments
    parser.add_argument('--use_pinn', action='store_true', help="Enable Physics-Informed loss term")
    parser.add_argument('--pinn_weight', type=float, default=1e-6, help="Weight for the PDE loss")
    parser.add_argument('--base_data_dir', type=str, default='data', help="Base directory for all data")
    
    args = parser.parse_args()
    main(args)