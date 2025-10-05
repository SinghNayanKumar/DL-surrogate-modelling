# --- START OF CORRECTED fine_tune_with_pinn.py ---
import torch
import argparse
import os
import glob
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb

# Ensure these imports match your project structure
from src.datasets.graph_dataset import IBeamGraphDataset
from src.models.gnn_variants import MPNN_Surrogate
from src.training.trainer import ModelTrainer

def fine_tune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. W&B Setup ---
    wandb.init(project="DL-surrogate-modelling", name=args.experiment_name, config=vars(args))

    # --- 2. Determine Model Architecture from Arguments ---
    # ### --- CHANGE START --- ###
    # Conditionally set the number of input features based on the new flag
    if args.use_one_hot:
        num_node_features = 16 # coords(3) + params(10) + one_hot(3)
        print("Initializing GNN for MULTI-MODAL (Generalist) fine-tuning with 16 input features.")
    else:
        num_node_features = 14 # coords(3) + params(11)
        print("Initializing GNN for UNI-MODAL (Specialist) fine-tuning with 14 input features.")
    # ### --- CHANGE END --- ###

    # --- 3. Load Data and Stats (Robustly) ---
    print(f"Loading stats from: {args.stats_y_path} and {args.stats_x_path}")
    stats_y = torch.load(args.stats_y_path)
    stats_x = torch.load(args.stats_x_path)
    stats = {
        'mean_y': stats_y['mean_y'], 'std_y': stats_y['std_y'],
        'mean_x': stats_x['mean_x'], 'std_x': stats_x['std_x']
    }
    
    print(f"Loading data from directory: {args.data_dir}")
    all_files = sorted(glob.glob(os.path.join(args.data_dir, '*.h5')))
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42)

    # ### --- CHANGE --- ### Pass the use_one_hot flag to the dataset
    train_dataset = IBeamGraphDataset(root_dir=args.data_dir, h5_file_list=train_files, stats=stats, use_one_hot=args.use_one_hot)
    val_dataset = IBeamGraphDataset(root_dir=args.data_dir, h5_file_list=val_files, stats=stats, use_one_hot=args.use_one_hot)
    train_loader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = PyGDataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
    print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # --- 4. Load PRE-TRAINED Data-Only Model ---
    # ### --- CHANGE --- ### Initialize the model with the correct number of features
    model = MPNN_Surrogate(node_in_features=num_node_features, node_out_features=3, hidden_size=args.hidden_size, use_pinn=True).to(device)
    
    print(f"Loading pre-trained data-only model from: {args.pretrained_checkpoint}")
    # This will now succeed because the model architecture matches the checkpoint
    model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=device))
    
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 5. Setup Optimizer for Fine-Tuning ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # --- 6. Run Fine-Tuning ---
    trainer_config = {
        'model_type': 'mpnn', 'use_pinn': True, 'pinn_weight': args.pinn_weight,
        'experiment_name': wandb.run.name, 'stats': stats,
        'pinn_warmup_epochs': args.pinn_warmup_epochs
    }
    trainer = ModelTrainer(model, train_loader, val_loader, optimizer, device, trainer_config, scheduler=scheduler)
    
    print("--- Starting Physics-Based Fine-Tuning ---")
    trainer.train(args.epochs)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Physics-Based Fine-Tuning of a Pre-trained GNN")
    
    parser.add_argument('--pretrained_checkpoint', type=str, required=True, help="Path to the .pth file of the pre-trained data-only model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the directory containing the .h5 data files.")
    parser.add_argument('--stats_x_path', type=str, required=True, help="Path to the correct input stats file (e.g., gnn_stats_x_scalar.pt or gnn_stats_x_onehot.pt).")
    parser.add_argument('--stats_y_path', type=str, required=True, help="Path to the gnn_stats_y.pt file.")
    
    parser.add_argument('--experiment_name', type=str, default="mpnn_pinn_finetune")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--pinn_weight', type=float, default=1.0) # Lowered default
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--pinn_warmup_epochs', type=int, default=0) # No warmup needed for fine-tuning
    
    # ### --- NEW ARGUMENT --- ###
    parser.add_argument('--use_one_hot', action='store_true', help="Set this flag if you are fine-tuning a multi-modal (one-hot) model.")
    
    args = parser.parse_args()
    fine_tune(args)

# --- END OF FILE ---