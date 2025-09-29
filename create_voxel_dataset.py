import os
import glob
import h5py
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

# --- Configuration ---
# This is the resolution of the output 3D "image". Adjust based on your GPU memory.
# The order is (Depth, Height, Width), corresponding to (Z, Y, X) axes.
# For an I-beam, Z (length) is typically the longest dimension.
VOXEL_RESOLUTION = (96, 48, 48)
INPUT_DIR = "data/h5_raw"  # Updated to match project structure
OUTPUT_DIR = "data/voxelized_big" # Updated to match project structure
INTERPOLATION_METHOD = 'linear' # 'linear' offers a good balance of speed and accuracy.

# =============================================================================
# Main Script Logic
# =============================================================================
def create_voxel_dataset():
    """
    Processes H5 files containing unstructured FEA mesh data and converts them
    into structured, voxelized 3D numpy arrays (.npz). This format is suitable
    for training Convolutional Neural Networks like U-Nets.

    For each simulation, this script generates:
    1. A multi-channel displacement field (dx, dy, dz).
    2. A binary mask defining the geometry of the object.
    3. An array of the scalar simulation parameters (e.g., force, material properties).
    """
    print("--- Starting Voxelization Process ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    h5_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.h5")))
    if not h5_files:
        print(f"Error: No H5 files found in '{INPUT_DIR}'. Please check the path.")
        return

    print(f"Found {len(h5_files)} H5 files to process.")

    # --- Step 1: Define a consistent interpolation grid ---
    # To ensure all voxel grids are comparable, we define a global bounding box.
    # We'll use the first file as a reference, assuming all simulations occupy
    # a similar spatial domain, which is a safe assumption for a DoE study.
    print("Defining a consistent grid from the first simulation file...")
    with h5py.File(h5_files[0], 'r') as f:
        coords = f['node_coordinates'][:]
        min_bounds = np.min(coords, axis=0)
        max_bounds = np.max(coords, axis=0)
        print(f"  - Min Bounds (X,Y,Z): {min_bounds}")
        print(f"  - Max Bounds (X,Y,Z): {max_bounds}")

    # Create the coordinate points for our target 3D grid
    # np.linspace creates evenly spaced points between the bounds.
    grid_x = np.linspace(min_bounds[0], max_bounds[0], VOXEL_RESOLUTION[2]) # Width
    grid_y = np.linspace(min_bounds[1], max_bounds[1], VOXEL_RESOLUTION[1]) # Height
    grid_z = np.linspace(min_bounds[2], max_bounds[2], VOXEL_RESOLUTION[0]) # Depth

    # np.meshgrid creates coordinate matrices from coordinate vectors.
    # 'ij' indexing results in a grid with shape (Z, Y, X) which matches our desired output.
    grid_Z, grid_Y, grid_X = np.meshgrid(grid_z, grid_y, grid_x, indexing='ij')
    
    # Flatten the grid coordinates into a list of points for griddata
    target_grid_points = np.vstack([grid_Z.ravel(), grid_Y.ravel(), grid_X.ravel()]).T

    # --- Step 2: Process each H5 file ---
    for h5_path in tqdm(h5_files, desc="Voxelizing Simulations"):
        try:
            with h5py.File(h5_path, 'r') as f:
                # Load the unstructured data from the H5 file
                points = f['node_coordinates'][:]           # Shape: (num_nodes, 3)
                displacement_values = f['displacement'][:]  # Shape: (num_nodes, 3)

                # --- 2a. Interpolate the displacement field onto the grid ---
                # griddata maps values from an unstructured point cloud (points)
                # to a structured grid (target_grid_points).
                grid_displacement = griddata(
                    points, 
                    displacement_values, 
                    target_grid_points, 
                    method=INTERPOLATION_METHOD,
                    fill_value=0.0 # Voxels outside the beam are filled with zero displacement
                )
                
                # Reshape the flattened array back to the 3D grid shape and reorder
                # from (D, H, W, 3) to the PyTorch standard (3, D, H, W) for channels-first.
                grid_displacement = grid_displacement.reshape(VOXEL_RESOLUTION[0], VOXEL_RESOLUTION[1], VOXEL_RESOLUTION[2], 3)
                grid_displacement_field = np.transpose(grid_displacement, (3, 0, 1, 2)) # (3, D, H, W)

                # --- 2b. Create a binary geometry mask ---
                # This tells the U-Net where the beam actually is.
                # We interpolate a field of '1s' from the node locations. Anything outside the
                # convex hull of the points will be filled with the fill_value (0.0).
                geom_values = np.ones(points.shape[0])
                grid_mask = griddata(
                    points,
                    geom_values,
                    target_grid_points,
                    method='nearest', # 'nearest' is fast and ideal for creating a sharp mask
                    fill_value=0.0
                )
                grid_mask = grid_mask.reshape(VOXEL_RESOLUTION)
                grid_mask = (grid_mask > 0.5).astype(np.uint8) # Ensure it's a binary 0 or 1 mask

                # --- 2c. Extract ALL simulation parameters from H5 attributes ---
                # This is the crucial step for conditioning the U-Net model.
                                
                # Define a mapping for categorical variables to numerical values
                load_type_map = {"bending_y": 0, "bending_x": 1, "torsion": 2}
                load_dist_map = {"uniform": 0, "linear_y": 1}

                # Extract continuous parameters
                beam_length = f.attrs.get('beam_length', 300.0)
                flange_width = f.attrs.get('flange_width', 100.0)
                flange_thickness = f.attrs.get('flange_thickness', 15.0)
                web_thickness = f.attrs.get('web_thickness', 10.0)
                beam_depth = f.attrs.get('beam_depth', 150.0)
                fillet_radius = f.attrs.get('fillet_radius', 12.0)
                youngs_modulus = f.attrs.get('youngs_modulus', 2.1e11)
                poissons_ratio = f.attrs.get('poissons_ratio', 0.3)
                force_magnitude = f.attrs.get('force_magnitude', 1500.0)

                # Extract and encode categorical parameters
                load_type_str = f.attrs.get('load_type', 'bending_y')
                load_dist_str = f.attrs.get('load_distribution', 'uniform')
                load_type_encoded = load_type_map.get(load_type_str, 0)
                load_dist_encoded = load_dist_map.get(load_dist_str, 0)
                
                # Assemble all parameters into a single NumPy array
                simulation_params = np.array([
                    beam_length, flange_width, flange_thickness, web_thickness,
                    beam_depth, fillet_radius, youngs_modulus, poissons_ratio,
                    force_magnitude, load_type_encoded, load_dist_encoded
                ], dtype=np.float32)

                # --- 2d. Save the processed arrays to a compressed NPZ file ---
                base_name = os.path.basename(h5_path).replace('.h5', '')
                output_path = os.path.join(OUTPUT_DIR, f"{base_name}.npz")
                
                np.savez_compressed(
                    output_path,
                    displacement_field=grid_displacement_field.astype(np.float32),
                    geometry=grid_mask,
                    simulation_params=simulation_params # Save the full parameters array
                )

        except Exception as e:
            print(f"\n[ERROR] Failed to process file {h5_path}: {e}")
            print("         Skipping this file.")

    print("\n--- Voxelization Complete ---")
    print(f"Processed data saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    create_voxel_dataset()