import pandas as pd
from evaluate.evaluate_model import evaluate_single_model 
import os

# =============================================================================
# CONFIGURING EXPERIMENTS HERE
# =============================================================================
CHECKPOINT_DIR = "checkpoints"
DATA_BASE_DIR = "data" 
STATS_BASE_DIR = "data" 
# --- NEW: Define a common output directory for visualization files ---
VISUALIZATION_OUTPUT_DIR = "results/for_figures"


EXPERIMENT_CONFIGS = [
    # ... (Configs for models you DON'T need to visualize remain unchanged) ...
    
    # ===================================================================
    # MODELS FOR "FIGURE 5: ARCHITECTURAL SHOWDOWN"
    # We need the best GNN vs. the U-Net on the same test samples.
    # ===================================================================
    {
        'model_name': 'Transformer - HIGH SIGNAL (Specialist)', 'model_type': 'transformer', 'batch_size': 1,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_transformer_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw_unimodal',
        'stats_y_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
        # --- ADDED FOR VISUALIZATION ---
        'save_for_visualization': True,
        'visualization_indices': [0, 3], # Save a couple of samples to choose the best one
        'visualization_output_dir': VISUALIZATION_OUTPUT_DIR
    },
    {
        'model_name': 'U-Net + Attn (High-Res) - HIGH SIGNAL (Specialist)', 'model_type': 'unet_small', 'batch_size': 1,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_unet_small_attention_data.pth', 
        'data_dir': f'{DATA_BASE_DIR}/voxelized_big_uni', 
        'stats_y_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/voxel_stats_big.pt', 
        'stats_x_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/param_stats_big.pt', 
        'use_attention': True,
        # --- ADDED FOR VISUALIZATION ---
        # NOTE: This part is tricky as GNNs and U-Nets use different data files. 
        # You will need to ensure that test sample #5 from the GNN data corresponds 
        # to test sample #5 from the U-Net data (which it should if the random_state is the same).
        'save_for_visualization': True,
        'visualization_indices': [0, 3], 
        'visualization_output_dir': VISUALIZATION_OUTPUT_DIR
    },

    # ===================================================================
    # MODELS FOR "FIGURE 7: THE PINN ADVANTAGE"
    # We need the baseline MPNN vs. the MPNN-PINN on the same test samples.
    # ===================================================================
    {
        'model_name': 'MPNN - HIGH SIGNAL (Specialist)', 'model_type': 'mpnn', 'batch_size': 1,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_mpnn_specialist_bending_y_data.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw_unimodal',
        'stats_y_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
        # --- ADDED FOR VISUALIZATION ---
        'save_for_visualization': True,
        'visualization_indices': [0, 3], # Use the same indices for a direct comparison
        'visualization_output_dir': VISUALIZATION_OUTPUT_DIR
    },
    {
        'model_name': 'MPNN PINN- HIGH SIGNAL (Specialist)', 'model_type': 'mpnn', 'batch_size': 1,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_mpnn_pinn_finetune_specialist.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_raw_unimodal',
        'stats_y_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/backup/Unimodal_stat/gnn_stats_x_scalar.pt',
        'ablation_one_hot': False,
        'use_pinn': True,
        'hidden_size': 128,
        # --- ADDED FOR VISUALIZATION ---
        'save_for_visualization': True,
        'visualization_indices': [0, 3], # Use the same indices for a direct comparison
        'visualization_output_dir': VISUALIZATION_OUTPUT_DIR
    },
    
    # ===================================================================
    # MODEL FOR "FIGURE 6: THE GENERALIST'S VERSATILITY"
    # We need the generalist model on a few different samples to find one of each load type.
    # ===================================================================
    {
        'model_name': 'MPNN - PINN LOW SIGNAL (Generalist)', 'model_type': 'mpnn', 'batch_size': 1,
        'checkpoint_path': f'{CHECKPOINT_DIR}/best_model_mpnn_pinn_finetune_generalist.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_multi_modal',
        'stats_y_path': f'{STATS_BASE_DIR}/backup/multi-modal_stat/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/backup/multi-modal_stat/gnn_stats_x_onehot.pt',
        'ablation_one_hot': True,
        'use_pinn': True,
        'hidden_size': 128,
        # --- ADDED FOR VISUALIZATION ---
        'save_for_visualization': True,
        'visualization_indices': list(range(25)), # Save a few to find good examples of each load type
        'visualization_output_dir': VISUALIZATION_OUTPUT_DIR
    },

    # --- (You can include all other configs here without the visualization keys) ---
    # Example:
    {
        'model_name': 'GCN - LOW SIGNAL (Generalist)', 'model_type': 'gcn', 'batch_size': 16,
        'checkpoint_path': f'{CHECKPOINT_DIR}/low_signal_models/best_model_gcn_data_onehot.pth',
        'data_dir': f'{DATA_BASE_DIR}/h5_multi_modal',
        'stats_y_path': f'{STATS_BASE_DIR}/backup/multi-modal_stat/gnn_stats_y.pt',
        'stats_x_path': f'{STATS_BASE_DIR}/backup/multi-modal_stat/gnn_stats_x_onehot.pt',
        'ablation_one_hot': True,
    },
    # ... etc for all other models ...
]


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    # The rest of this script remains exactly the same
    all_results = []
    
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        print("\n" + "="*50)
        print(f"  Running Evaluation {i+1}/{len(EXPERIMENT_CONFIGS)}: {config['model_name']}")
        print("="*50)
        
        try:
            # The same function call will now check for the visualization keys internally
            results = evaluate_single_model(config)
            results['Model'] = config['model_name']
            all_results.append(results)
        except FileNotFoundError as e:
            print(f"\n!!! ERROR for model '{config['model_name']}' !!!")
            print(f"  Could not find a required file: {e}")
            print("  Please check the paths in the EXPERIMENT_CONFIGS list.")
            print("  Skipping this model.\n")
        except Exception as e:
            print(f"\n!!! An unexpected error occurred for model '{config['model_name']}': {e} !!!")
            print("  Skipping this model.\n")

    if all_results:
        df = pd.DataFrame(all_results)
        column_order = ['Model', 'mae_mm', 'rl2_percent', 'r2_score', 'inference_ms', 'params_M']
        df = df[column_order]
        df.rename(columns={
            'mae_mm': 'MAE (mm) ↓', 'rl2_percent': 'R-L2 (%) ↓', 'r2_score': 'R² Score ↑',
            'inference_ms': 'Inference (ms) ↓', 'params_M': 'Params (M) ↓'
        }, inplace=True)
        df.set_index('Model', inplace=True)
        
        print("\n\n" + "="*80)
        print("                       --- FINAL EVALUATION RESULTS ---")
        print("="*80)
        print(df.to_string(float_format="%.4f"))
        print("="*80)
        
        output_csv_path = "results/evaluation_results/final_comparison_table.csv"
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path)
        print(f"\nResults table saved to {output_csv_path}")