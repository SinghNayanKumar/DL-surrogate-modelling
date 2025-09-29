import os
import glob
import h5py
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

# --- Configuration ---
# This is the resolution of the output 3D "image". Adjust based on your GPU memory.
# (Depth, Height, Width) - Note: Z is the longest axis (beam_length)
VOXEL_RESOLUTION = (128, 64, 64) 
INPUT_DIR = "output_data_3d"
OUTPUT_DIR = "output_voxel_data"
INTERPOLATION_METHOD = 'linear' # 'linear' is good, 'nearest' is faster but less accurate

# =============================================================================
# Main Script Logic
# =============================================================================
def create_voxel_dataset():
    """
    Processes H5 files of unstructured mesh data and converts them into
    voxelized 3D numpy arrays suitable for U-Nets/CNNs.
    """
    print("--- Starting Voxelization Process ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    h5_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.h5")))
    if not h5_files:
        print(f"Error: No H5 files found in '{INPUT_DIR}'. Please generate data first.")
        return

    print(f"Found {len(h5_files)} files to process.")

    # Determine the global bounding box from the first file to create a consistent grid
    # This assumes all beams occupy a similar space, which is true for our DoE.
    with h5py.File(h5_files[0], 'r') as f:
        coords = f['node_coordinates'][:]
        min_bounds = np.min(coords, axis=0)
        max_bounds = np.max(coords, axis=0)

    # Create the target grid coordinates
    grid_x = np.linspace(min_bounds[0], max_bounds[0], VOXEL_RESOLUTION[2])
    grid_y = np.linspace(min_bounds[1], max_bounds[1], VOXEL_RESOLUTION[1])
    grid_z = np.linspace(min_bounds[2], max_bounds[2], VOXEL_RESOLUTION[0])
    grid_X, grid_Y, grid_Z = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    target_grid_points = np.vstack([grid_X.ravel(), grid_Y.ravel(), grid_Z.ravel()]).T

    # Process each file
    for h5_path in tqdm(h5_files, desc="Voxelizing Simulations"):
        try:
            with h5py.File(h5_path, 'r') as f:
                # Load the unstructured data
                points = f['node_coordinates'][:]
                displacement_values = f['displacement'][:]

                # --- 1. Interpolate the displacement field ---
                grid_displacement = griddata(
                    points, 
                    displacement_values, 
                    target_grid_points, 
                    method=INTERPOLATION_METHOD,
                    fill_value=0.0 # Fill outside the beam with zero displacement
                )
                # Reshape the flattened array back to the 3D grid shape
                grid_displacement = grid_displacement.reshape(VOXEL_RESOLUTION[0], VOXEL_RESOLUTION[1], VOXEL_RESOLUTION[2], 3)

                # --- 2. Create a binary geometry mask ---
                # This tells the U-Net where the beam actually is.
                # We interpolate a field of '1s' and anything outside the convex hull will be the fill_value.
                geom_values = np.ones(points.shape[0])
                grid_mask = griddata(
                    points,
                    geom_values,
                    target_grid_points,
                    method='nearest', # Nearest is fast and good enough for a mask
                    fill_value=0.0
                )
                grid_mask = grid_mask.reshape(VOXEL_RESOLUTION)
                grid_mask = (grid_mask > 0.5).astype(np.uint8) # Convert to binary mask

                # --- 3. Save the processed arrays ---
                base_name = os.path.basename(h5_path).replace('.h5', '')
                
                # Save as a single compressed file for convenience
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}_voxelized.npz")
                np.savez_compressed(
                    output_path,
                    displacement=grid_displacement.astype(np.float32),
                    mask=grid_mask
                )

        except Exception as e:
            print(f"Error processing file {h5_path}: {e}")

    print("\n--- Voxelization Complete ---")
    print(f"Processed data saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    create_voxel_dataset()