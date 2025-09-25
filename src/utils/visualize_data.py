import os
import sys
import glob
import torch
import pyvista as pv
import numpy as np
import h5py

# The '..' goes up one level, so we go up two levels to reach the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

# Correctly import from the new modular structure
from src.datasets.graph_dataset import IBeamGraphDataset

# --- Configuration ---
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, 'data', 'h5_raw') # The directory containing your .h5 files

def main():
    """
    Main function to load a graph from the dataset and visualize it.
    This script serves as a sanity check to ensure the data loader is
    parsing the raw H5 files correctly before starting a training run.
    """
    print(f"Searching for .h5 files in '{DATA_DIRECTORY}'...")
    
    # --- Step 1: Find all data files ---
    # In the new design, we manually find the files and pass the list to the dataset.
    # This allows for easy train/val/test splitting in the main training script.
    # For this visualization script, we'll just grab all of them.
    all_h5_files = sorted(glob.glob(os.path.join(DATA_DIRECTORY, '*.h5')))

    if not all_h5_files:
        print(f"Error: No .h5 files found in '{DATA_DIRECTORY}'. Please check the path.")
        return

    print(f"Found {len(all_h5_files)} simulation files.")

    # --- Step 2: Initialize the Dataset ---
    # Pass the discovered file list to the dataset constructor.
    print("Initializing IBeamGraphDataset...")
    try:
        # The 'root_dir' is still used by PyG for potential processing/caching,
        # and 'h5_file_list' tells the dataset which specific files to load.
        dataset = IBeamGraphDataset(root_dir=DATA_DIRECTORY, h5_file_list=all_h5_files)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        return

    print(f"Dataset loaded successfully with {len(dataset)} samples.")

    # --- Step 3: Load a single sample ---
    sample_index = 0
    print(f"\nLoading sample {sample_index} from file: {os.path.basename(dataset.h5_files[sample_index])}...")
    graph_data = dataset.get(sample_index)
    print("\n--- PyG Data Object ---")
    print(graph_data)
    print("-------------------------")

    # --- Step 4: Prepare data for PyVista ---
    points = graph_data.pos.numpy()
    
    # To render the volume, we need the original tetrahedral topology.
    # We can get the path from the dataset object itself.
    h5_path = dataset.h5_files[sample_index]
    with h5py.File(h5_path, 'r') as f:
        topology = f['topology'][:]

    # PyVista's UnstructuredGrid requires a specific cell format:
    # [num_points_in_cell, point_idx_1, point_idx_2, ...]
    cells = topology

    # We need to calculate the number of tetrahedra correctly from the 1D array size.
    # Each tetrahedron takes 5 slots in the array (1 for the '4', 4 for the nodes).
    num_tetra = topology.size // 5
    cell_types = np.full(num_tetra, pv.CellType.TETRA, dtype=np.uint8)

    
    # Create the PyVista UnstructuredGrid object
    mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # --- Step 5: Add data to the mesh for visualization ---
    # Calculate the magnitude of the displacement vector for scalar coloring
    displacement_magnitude = torch.norm(graph_data.y, p=2, dim=-1).numpy()
    mesh['displacement_magnitude'] = displacement_magnitude

    # Create the deformed mesh by adding the displacement vectors to the original nodes
    # We scale the displacement for better visibility in the plot
    deformation_scale_factor = 50 
    deformed_points = graph_data.pos + (graph_data.y * deformation_scale_factor)
    deformed_mesh = mesh.copy()
    deformed_mesh.points = deformed_points.numpy()

    # --- Step 6: Plotting ---
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800]) # Create a 1x2 grid for side-by-side plots

    # Plot 1: Original mesh with displacement coloring
    plotter.subplot(0, 0)
    plotter.add_text("Displacement Magnitude on Original Mesh", font_size=12)
    sargs = dict(title='Displacement (m)') # Scalar bar arguments
    plotter.add_mesh(mesh, scalars='displacement_magnitude', cmap='viridis', 
                     show_edges=True, edge_color='grey', scalar_bar_args=sargs)
    plotter.view_isometric()
    
    # Plot 2: Deformed mesh (scaled for visibility)
    plotter.subplot(0, 1)
    plotter.add_text(f"Deformed Mesh ({deformation_scale_factor}x Scale)", font_size=12)
    plotter.add_mesh(deformed_mesh, scalars='displacement_magnitude', cmap='viridis',
                     show_edges=True, edge_color='grey', scalar_bar_args=sargs)
    plotter.view_isometric()

    # Link cameras for easy comparison
    plotter.link_views() 
    
    print("\nShowing plot. Close the window to exit.")
    plotter.show()


if __name__ == '__main__':
    main()