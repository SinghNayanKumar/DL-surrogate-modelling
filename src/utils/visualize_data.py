import torch
import pyvista as pv
import numpy as np
from src.datasets.graph_dataset import IBeamGraphDataset

# --- Configuration ---
DATA_DIRECTORY = 'data/h5_raw' # The directory containing your .h5 files

def main():
    """
    Main function to load a graph and visualize it.
    """
    print("Initializing dataset...")
    try:
        dataset = IBeamGraphDataset(root_dir=DATA_DIRECTORY)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print(f"Please ensure the '{DATA_DIRECTORY}' directory exists and contains .h5 files.")
        return

    if len(dataset) == 0:
        print(f"No .h5 files found in '{DATA_DIRECTORY}'. Exiting.")
        return

    print(f"Dataset loaded successfully with {len(dataset)} samples.")

    # --- Load a single sample from the dataset ---
    sample_index = 0
    print(f"\nLoading sample {sample_index}...")
    graph_data = dataset.get(sample_index)
    print("\n--- Graph Data Object ---")
    print(graph_data)
    print("-------------------------")

    # --- Prepare data for PyVista ---
    # PyVista needs NumPy arrays for points and cell topology
    points = graph_data.pos.numpy()
    
    # We need the original tetrahedron topology for volumetric rendering.
    # We can re-load it or, for simplicity, assume it can be accessed.
    # Let's re-load it for this standalone script.
    h5_path = dataset.h5_files[sample_index]
    import h5py
    with h5py.File(h5_path, 'r') as f:
        topology = f['topology'][:]

    # PyVista's UnstructuredGrid requires a specific cell format:
    # [num_points_in_cell, point_idx_1, point_idx_2, ...]
    # For tetrahedrons, num_points_in_cell is 4.
    num_tetra = topology.shape[0]
    cells = np.hstack([np.full((num_tetra, 1), 4), topology]).flatten()
    cell_types = np.full(num_tetra, pv.CellType.TETRA)
    
    # Create the PyVista UnstructuredGrid object
    mesh = pv.UnstructuredGrid(cells, cell_types, points)

    # --- Add data to the mesh for visualization ---
    # Calculate the magnitude of the displacement vector for scalar coloring
    displacement_magnitude = torch.norm(graph_data.y, p=2, dim=-1).numpy()
    mesh['displacement_magnitude'] = displacement_magnitude

    # Also add the deformed mesh for comparison
    deformed_points = graph_data.pos + graph_data.y
    deformed_mesh = mesh.copy()
    deformed_mesh.points = deformed_points.numpy()

    # --- Plotting ---
    plotter = pv.Plotter(shape=(1, 2)) # Create a 1x2 grid for side-by-side plots

    # Plot 1: Original mesh with displacement coloring
    plotter.subplot(0, 0)
    plotter.add_text("Displacement Magnitude on Original Mesh", font_size=12)
    plotter.add_mesh(mesh, scalars='displacement_magnitude', cmap='viridis', 
                     show_edges=True, edge_color='grey')
    
    # Plot 2: Deformed mesh (scaled for visibility)
    plotter.subplot(0, 1)
    plotter.add_text("Deformed Mesh (scaled)", font_size=12)
    plotter.add_mesh(deformed_mesh, scalars='displacement_magnitude', cmap='viridis',
                     show_edges=True, edge_color='grey')
    plotter.link_views() # Link cameras for easy comparison
    
    print("\nShowing plot. Close the window to exit.")
    plotter.show()


if __name__ == '__main__':
    main()