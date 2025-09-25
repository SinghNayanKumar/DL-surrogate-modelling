import os
import glob
import torch
import argparse
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader

# Import all our custom modules
from datasets.graph_dataset import IBeamGraphDataset
from datasets.voxel_dataset import IBeamVoxelDataset
from models.gnn_variants import GCN_Surrogate, GAT_Surrogate, MPNN_Surrogate
from models.graph_transformer import GraphTransformer_Surrogate
from models.unet_variants import UNet3D
from training.trainer import ModelTrainer

MODEL_MAPPING = {
    'gcn': GCN_Surrogate, 'gat': GAT_Surrogate, 'mpnn': MPNN_Surrogate,
    'transformer': GraphTransformer_Surrogate, 'unet': UNet3D,
}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    is_gnn = args.model_type in ['gcn', 'gat', 'mpnn', 'transformer']
    if is_gnn:
        all_files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    else: # unet
        all_files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42) # 0.125 * 0.8 = 0.1

    if is_gnn:
        train_dataset = IBeamGraphDataset(root_dir=args.data_dir, h5_file_list=train_files)
        val_dataset = IBeamGraphDataset(root_dir=args.data_dir, h5_file_list=val_files)
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size)
    else:
        train_dataset = IBeamVoxelDataset(file_list=train_files)
        val_dataset = IBeamVoxelDataset(file_list=val_files)
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size)
        
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
    
    args = parser.parse_args()
    main(args)