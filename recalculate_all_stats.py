import os
import glob
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the functions from your existing stats.py file
from src.utils.stats import (
    compute_and_save_stats,
    compute_and_save_graph_input_stats,
    compute_and_save_voxel_stats,
    compute_and_save_param_stats
)

# =============================================================================
# SCRIPT CONFIGURATION - SINGLE SOURCE OF TRUTH
# =============================================================================
DATA_BASE_DIR = "data"

# --- Define the MASTER source of all simulation data ---
# This is the single source of truth for our data split.
MASTER_H5_DIR = os.path.join(DATA_BASE_DIR, "h5_raw")

# --- Define directories for derived data types ---
# The script will map files from the master split to these directories.
VOXEL_BIG_DIR = os.path.join(DATA_BASE_DIR, "voxelized_big")

# --- Define the EXACT output paths for the new, clean stats files ---
STATS_Y_GNN_PATH = os.path.join(DATA_BASE_DIR, "gnn_stats_y.pt")
STATS_X_GNN_ONEHOT_PATH = os.path.join(DATA_BASE_DIR, "gnn_stats_x_onehot.pt")
STATS_X_GNN_SCALAR_PATH = os.path.join(DATA_BASE_DIR, "gnn_stats_x_scalar.pt")

STATS_Y_VOXEL_BIG_PATH = os.path.join(DATA_BASE_DIR, "voxel_stats_big.pt")
STATS_X_VOXEL_BIG_PATH = os.path.join(DATA_BASE_DIR, "param_stats_big.pt")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("--- Starting Clean Recalculation of All Normalization Statistics ---")

    # 1. SAFETY STEP: Delete all old stats files.
    print("\n--- Step 1: Deleting old statistics files... ---")
    files_to_delete = [
        STATS_Y_GNN_PATH, STATS_X_GNN_ONEHOT_PATH, STATS_X_GNN_SCALAR_PATH,
        STATS_Y_VOXEL_BIG_PATH, STATS_X_VOXEL_BIG_PATH
    ]
    for f_path in files_to_delete:
        try:
            os.remove(f_path)
            print(f"  - Deleted: {f_path}")
        except FileNotFoundError:
            print(f"  - Not found, skipping: {f_path}")

    # 2. CREATE THE CANONICAL DATA SPLIT (THE MOST IMPORTANT STEP)
    print(f"\n--- Step 2: Creating the canonical train/test split from '{MASTER_H5_DIR}' ---")
    all_master_files = sorted(glob.glob(os.path.join(MASTER_H5_DIR, '*.h5')))
    if not all_master_files:
        raise FileNotFoundError(f"CRITICAL ERROR: No .h5 files found in the master directory: '{MASTER_H5_DIR}'")

    # This is the ONE AND ONLY time we call train_test_split.
    # This `canonical_train_files` list is now sacred.
    canonical_train_files, _ = train_test_split(all_master_files, test_size=0.2, random_state=42)
    print(f"  - Canonical split created. Training set size: {len(canonical_train_files)} files.")

    # 3. Calculate all GNN stats based on the canonical split.
    print("\n--- Step 3: Calculating GNN Statistics from the canonical split ---")

    # GNN Target Stats (Y) - calculated from the full training set.
    print("  - Calculating target stats (Y) for GNNs...")
    compute_and_save_stats(canonical_train_files, STATS_Y_GNN_PATH)

    # GNN Input Stats (X) for Generalist (One-Hot) - uses the full training set.
    print("  - Calculating one-hot input stats (X) for Generalist GNNs...")
    compute_and_save_graph_input_stats(canonical_train_files, STATS_X_GNN_ONEHOT_PATH, use_one_hot=True)

    # GNN Input Stats (X) for Specialist (Scalar) - filter the canonical training set.
    print("  - Calculating scalar input stats (X) for Specialist GNNs...")
    unimodal_train_files = []
    for h5_path in tqdm(canonical_train_files, desc="Filtering for unimodal"):
        with h5py.File(h5_path, 'r') as f:
            if f.attrs.get('load_type') == 'bending_y':
                unimodal_train_files.append(h5_path)
    
    if not unimodal_train_files:
         raise ValueError("CRITICAL ERROR: No 'bending_y' files found in the canonical training split.")
    
    compute_and_save_graph_input_stats(unimodal_train_files, STATS_X_GNN_SCALAR_PATH, use_one_hot=False)

    # 4. Calculate Voxel stats based on the canonical split.
    print("\n--- Step 4: Calculating Voxel Statistics from the canonical split ---")
    
    # Map the canonical H5 training files to their corresponding NPZ voxel files.
    voxel_train_files = []
    for h5_path in canonical_train_files:
        base_name = os.path.basename(h5_path).replace('.h5', '.npz')
        voxel_path = os.path.join(VOXEL_BIG_DIR, base_name)
        if os.path.exists(voxel_path):
            voxel_train_files.append(voxel_path)
        else:
            print(f"  - Warning: Corresponding voxel file not found for {h5_path}, skipping.")

    if len(voxel_train_files) != len(canonical_train_files):
        print("  - Warning: Mismatch in number of H5 and Voxel files. Ensure all data is processed.")

    if not voxel_train_files:
         raise FileNotFoundError(f"CRITICAL ERROR: No corresponding NPZ files found in '{VOXEL_BIG_DIR}' for the training split.")

    print(f"  - Found {len(voxel_train_files)} corresponding voxel files for stats calculation.")
    compute_and_save_voxel_stats(voxel_train_files, STATS_Y_VOXEL_BIG_PATH)
    compute_and_save_param_stats(voxel_train_files, STATS_X_VOXEL_BIG_PATH)

    print("\n--- All statistics have been successfully and consistently recalculated! ---")
    print("You are now ready to run your training suite.")