import os
import glob
import torch
import argparse
import wandb
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datasets.graph_dataset import IBeamGraphDataset
from src.datasets.voxel_dataset import IBeamVoxelDataset
from src.models.gnn_variants import GCN_Surrogate, GAT_Surrogate, MPNN_Surrogate
from src.models.graph_transformer import GraphTransformer_Surrogate
from src.models.unet_variants import UNet3D, UNet3D_Small
from src.training.trainer import ModelTrainer
from src.utils.stats import compute_and_save_stats, compute_and_save_voxel_stats, compute_and_save_param_stats, compute_and_save_graph_input_stats


MODEL_MAPPING = {
    'gcn': GCN_Surrogate, 'gat': GAT_Surrogate, 'mpnn': MPNN_Surrogate,
    'transformer': GraphTransformer_Surrogate, 'unet': UNet3D,'unet_small': UNet3D_Small,
}

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    is_gnn = args.model_type in ['gcn', 'gat', 'mpnn', 'transformer']

    # --- W&B Naming ---
    base_name = args.model_type
    if args.experiment_suffix:
        base_name += f"_{args.experiment_suffix}"
    
    experiment_name = f"{base_name}_data"
    if is_gnn and args.ablation_one_hot:
        experiment_name += "_onehot"
    
    wandb.init(project="DL-surrogate-modelling", name=experiment_name, config=vars(args))

    # --- Data Loading and Splitting ---
    # ### --- CHANGE --- ### Use the specified data_dir
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The specified data directory does not exist: {data_dir}")
        
    data_ext = '*.h5' if is_gnn else '*.npz'
    all_files = sorted(glob.glob(os.path.join(data_dir, data_ext)))
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

    # --- Normalization Stats ---
    stats = {}
    if is_gnn:
        # === REPLACE THE OLD BLOCK WITH THIS ===
        stats_y_path = os.path.join('data', 'gnn_stats_y.pt')
        stats_x_suffix = "_onehot" if args.ablation_one_hot else "_scalar"
        stats_x_path = os.path.join('data', f'gnn_stats_x{stats_x_suffix}.pt')

        try:
            print(f"Loading pre-computed stats from {stats_x_path} and {stats_y_path}")
            stats_y = torch.load(stats_y_path)
            stats_x = torch.load(stats_x_path)
            stats = {'mean_y': stats_y['mean_y'], 'std_y': stats_y['std_y'],
                    'mean_x': stats_x['mean_x'], 'std_x': stats_x['std_x']}
        except FileNotFoundError:
            print("!!! FATAL ERROR: Statistics files not found. !!!")
            print("-> Please run a dedicated script (like 'recalculate_all_stats.py') to generate them once before training.")
            exit()

    else:
        # U-Net stats logic remains the same
        voxel_stats_path = os.path.join('data', 'voxel_stats.pt')
        param_stats_path = os.path.join('data', 'param_stats.pt')
        mean_y, std_y = compute_and_save_voxel_stats(train_files, voxel_stats_path)
        mean_x, std_x = compute_and_save_param_stats(train_files, param_stats_path)
        stats = {'mean_y': mean_y, 'std_y': std_y, 'mean_x_params': mean_x, 'std_x_params': std_x}

    # --- Dataset and DataLoader ---
    if is_gnn:
        # ### --- CHANGE --- ### Pass the flag to the dataset
        train_dataset = IBeamGraphDataset(root_dir=data_dir, h5_file_list=train_files, stats=stats, use_one_hot=args.ablation_one_hot)
        val_dataset = IBeamGraphDataset(root_dir=data_dir, h5_file_list=val_files, stats=stats, use_one_hot=args.ablation_one_hot)
        train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    else:
        train_dataset = IBeamVoxelDataset(file_list=train_files, stats=stats)
        val_dataset = IBeamVoxelDataset(file_list=val_files, stats=stats)
        train_loader = TorchDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = TorchDataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        
    print(f"Data loaded from '{data_dir}': {len(train_dataset)} train, {len(val_dataset)} val, {len(test_files)} test samples.")

    # --- Model Selection ---
    model_class = MODEL_MAPPING[args.model_type]
    if is_gnn:
        # ### --- CHANGE --- ### Conditional feature size
        if args.ablation_one_hot:
            num_node_features = 16 # coords(3) + params(10) + one_hot(3)
            print("Initializing GNN for MULTI-MODAL (Generalist) training with 16 input features.")
        else:
            num_node_features = 14 # coords(3) + params(11)
            print("Initializing GNN for UNIMODAL (Specialist) training with 14 input features.")
        
        model = model_class(node_in_features=num_node_features, node_out_features=3, hidden_size=args.hidden_size).to(device)
    else:
        # U-Net logic remains the same for now
        num_input_channels = 1 + 11 # geometry + params
        model = model_class(in_channels=num_input_channels, out_channels=3, use_attention=args.use_attention).to(device)
    
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

    trainer_config = {'model_type': args.model_type, 'use_pinn': args.use_pinn, 'pinn_weight': args.pinn_weight, 'experiment_name': wandb.run.name}
    trainer = ModelTrainer(model, train_loader, val_loader, optimizer, device, trainer_config, scheduler=scheduler)
    trainer.train(args.epochs)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FEA Surrogate Model Training")
    parser.add_argument('--model_type', type=str, required=True, choices=MODEL_MAPPING.keys())
    # ### --- CHANGE --- ### Added data_dir argument
    parser.add_argument('--data_dir', type=str, default='data/h5_multi_modal', help="Directory containing the training data (.h5 or .npz files)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--use_pinn', action='store_true')
    parser.add_argument('--pinn_weight', type=float, default=1e-6)
    parser.add_argument('--ablation_one_hot', action='store_true', help="Use one-hot encoding for load_type. MUST be used for multi-modal training.")
    # ### --- CHANGE --- ### Added for better naming in W&B
    parser.add_argument('--experiment_suffix', type=str, default="", help="Optional suffix for the W&B experiment name (e.g., 'unimodal')")
    
    args = parser.parse_args()
    main(args)