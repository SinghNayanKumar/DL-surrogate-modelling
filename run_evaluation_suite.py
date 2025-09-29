import pandas as pd
from evaluate.evaluate_model import evaluate_single_model 
import os

# =============================================================================
# CONFIGURING EXPERIMENTS HERE
# =============================================================================
# Base paths - PLEASE VERIFY THESE ARE CORRECT FOR YOUR SYSTEM
CHECKPOINT_DIR = "checkpoints"
DATA_BASE_DIR = "data" # Assumes 'data/h5_multi_modal', 'data/h5_unimodal_bending_y', etc. exist
STATS_BASE_DIR = "data" # Assumes your .pt stats files are in the 'data' directory

# This list defines every model that will be evaluated.
# Each dictionary contains all the info needed to run one evaluation.
EXPERIMENT_CONFIGS = [
    # --- GNN Generalists (Multi-modal) ---



    # --- GNN Specialists (Unimodal) ---
    {
        'model_name': 'GCN (Specialist)', 'model_type': 'gcn', 'batch_size': 16,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_gcn_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw',
        'stats_y_path': f'{STATS_BASE_DIR}/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
    },
    {
        'model_name': 'GAT (Specialist)', 'model_type': 'gat', 'batch_size': 16,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_gat_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw',
        'stats_y_path': f'{STATS_BASE_DIR}/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
    },
    {
        'model_name': 'MPNN (Specialist)', 'model_type': 'mpnn', 'batch_size': 16,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_mpnn_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw',
        'stats_y_path': f'{STATS_BASE_DIR}/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
    },
        {
        'model_name': 'Transformer (specialist)', 'model_type': 'transformer', 'batch_size': 16,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_transformer_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw',
        'stats_y_path': f'{STATS_BASE_DIR}/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
    },
    
    # --- U-Net Baselines ---

    {
        'model_name': 'U-Net Small + Attn (High-Res)', 'model_type': 'unet_small', 'batch_size': 8,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_unet_small_attention_data.pth', 
        'data_dir': f'{DATA_BASE_DIR}/voxelized_big', 
        'stats_y_path': f'{STATS_BASE_DIR}/voxel_stats_big.pt', 
        'stats_x_path': f'{STATS_BASE_DIR}/param_stats_big.pt', 
        'use_attention': True
    },
]


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    all_results = []
    
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        print("\n" + "="*50)
        print(f"  Running Evaluation {i+1}/{len(EXPERIMENT_CONFIGS)}: {config['model_name']}")
        print("="*50)
        
        try:
            results = evaluate_single_model(config)
            results['Model'] = config['model_name'] # Add model name for the table
            all_results.append(results)
        except FileNotFoundError as e:
            print(f"\n!!! ERROR for model '{config['model_name']}' !!!")
            print(f"  Could not find a required file: {e}")
            print("  Please check the paths in the EXPERIMENT_CONFIGS list.")
            print("  Skipping this model.\n")
        except Exception as e:
            print(f"\n!!! An unexpected error occurred for model '{config['model_name']}': {e} !!!")
            print("  Skipping this model.\n")

    # --- Display Final Results Table ---
    if all_results:
        # Create a pandas DataFrame for beautiful printing
        df = pd.DataFrame(all_results)
        
        # Reorder columns for clarity
        column_order = [
            'Model', 'mae_mm', 'rl2_percent', 'r2_score', 'inference_ms', 'params_M'
        ]
        df = df[column_order]
        
        # Rename columns for the final table
        df.rename(columns={
            'mae_mm': 'MAE (mm) ↓',
            'rl2_percent': 'R-L2 (%) ↓',
            'r2_score': 'R² Score ↑',
            'inference_ms': 'Inference (ms) ↓',
            'params_M': 'Params (M) ↓'
        }, inplace=True)
        
        # Set Model as index for better readability
        df.set_index('Model', inplace=True)
        
        print("\n\n" + "="*80)
        print("                       --- FINAL EVALUATION RESULTS ---")
        print("="*80)
        print(df.to_string(float_format="%.4f"))
        print("="*80)
        
        # Save the results to a CSV file for your paper
        output_csv_path = "results/evaluation_results/final_comparison_table.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path)
        print(f"\nResults table saved to {output_csv_path}")