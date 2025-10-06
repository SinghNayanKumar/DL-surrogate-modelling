import os
import argparse
import pyvista as pv
import numpy as np
import h5py

def get_h5_data(h5_path):
    """Loads topology, bounds, and cell info from an H5 file."""
    with h5py.File(h5_path, 'r') as f:
        topology, coords = f['topology'][:], f['node_coordinates'][:]
        min_bounds, max_bounds = np.min(coords, axis=0), np.max(coords, axis=0)
    
    if topology.ndim == 1 and topology.size % 5 == 0:
        num_tetra = topology.size // 5
        cells = topology.reshape((num_tetra, 5))
    else:
        num_tetra = topology.shape[0]
        cells = np.full((num_tetra, 5), 4, dtype=topology.dtype); cells[:, 1:] = topology
        
    cell_types = np.full(num_tetra, pv.CellType.TETRA, dtype=np.uint8)
    return cells.flatten(), cell_types, min_bounds, max_bounds, coords

def generate_plot(mesh, title, scalars_key, clim, cmap, sargs, output_path, warp_factor=1.0, warp_key='displacement'):
    """Generates and saves a single, high-quality plot with optional deformation."""
    plotter = pv.Plotter(window_size=[800, 800], off_screen=True)
    plotter.add_text(title, font_size=12, position='upper_edge', color='black')
    
    if warp_factor > 1.0 and warp_key in mesh.point_data:
        warped_mesh = mesh.warp_by_vector(warp_key, factor=warp_factor)
        plotter.add_mesh(warped_mesh, scalars=scalars_key, clim=clim, cmap=cmap, scalar_bar_args=sargs)
        plotter.add_mesh(mesh, style='wireframe', color='grey', opacity=0.2)
    else:
        plotter.add_mesh(mesh, scalars=scalars_key, clim=clim, cmap=cmap, scalar_bar_args=sargs, show_edges=False)

    plotter.view_isometric()
    plotter.background_color = 'white'
    print(f"  - Saving individual plot: {os.path.basename(output_path)}")
    plotter.screenshot(output_path)
    plotter.close()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_meshes = {}
    all_gt_magnitudes = []
    gt_mesh = None

    # --- 1. Process all input files and prepare mesh objects ---
    for file_path, name in zip(args.files, args.names):
        data = np.load(file_path)
        
        # --- Process GNN Files ---
        if 'U-Net' not in name:
            h5_path = str(data['original_h5_path'])
            cells, cell_types, _, _, coords = get_h5_data(h5_path)
            mesh = pv.UnstructuredGrid(cells, cell_types, coords)
            
            mesh.point_data['displacement'] = data['prediction']
            mesh.point_data['magnitude'] = np.linalg.norm(data['prediction'], axis=1)
            mesh.point_data['error'] = np.linalg.norm(data['prediction'] - data['ground_truth'], axis=1)
            all_meshes[name] = mesh

            if gt_mesh is None:
                gt_mesh = mesh.copy()
                gt_mesh.point_data['displacement'] = data['ground_truth']
                gt_mesh.point_data['magnitude'] = np.linalg.norm(data['ground_truth'], axis=1)
                all_gt_magnitudes.append(gt_mesh.point_data['magnitude'])

        # --- Process U-Net Files ---
        else:
            # This is the robust method for handling voxel data. It creates a full grid,
            # assigns all data arrays, and then uses a threshold filter to cleanly
            # extract the geometry defined by the mask.
            h5_path = str(data['original_h5_path'])
            _, _, min_bounds, max_bounds, _ = get_h5_data(h5_path)
            
            mask_3d = data['geometry_mask']
            pred_vectors = np.transpose(data['prediction'], (1, 2, 3, 0)) # D, H, W, C
            gt_vectors = np.transpose(data['ground_truth'], (1, 2, 3, 0))   # D, H, W, C
            
            depth, height, width = mask_3d.shape
            grid_dims = (width, height, depth)
            grid = pv.ImageData(
                dimensions=grid_dims,
                origin=min_bounds,
                spacing=(max_bounds - min_bounds) / (np.array(grid_dims) - 1)
            )
            
            # Assign data arrays to the grid's points. The 'F' (Fortran) order is
            # crucial for correctly mapping NumPy's (D,H,W) arrays to PyVista's (X,Y,Z) grid.
            grid.point_data['mask'] = mask_3d.flatten(order='F')
            
            displacement = np.stack([pred_vectors[..., i].flatten(order='F') for i in range(3)], axis=1)
            gt_displacement = np.stack([gt_vectors[..., i].flatten(order='F') for i in range(3)], axis=1)
            
            grid.point_data['displacement'] = displacement
            grid.point_data['magnitude'] = np.linalg.norm(displacement, axis=1)
            grid.point_data['error'] = np.linalg.norm(displacement - gt_displacement, axis=1)
            
            # Extract the clean geometry from the grid using the mask.
            mesh = grid.threshold(value=0.1, scalars='mask')
            
            if mesh.n_points == 0:
                print(f"!!! WARNING for {name}: Thresholding resulted in an empty mesh. The mask contains no values > 0.1.")
            
            all_meshes[name] = mesh

    if not all_meshes or gt_mesh is None:
        print("Error: Could not create any valid meshes for plotting. Please check input files.")
        return

    # --- 2. Define Consistent Color Ranges ---
    global_max_magnitude = max(np.max(mags) for mags in all_gt_magnitudes)
    mag_clim = [0, global_max_magnitude]
    
    max_error = max(mesh['error'].max() for mesh in all_meshes.values() if mesh.n_points > 0)
    error_clim = [0, max_error]
    sargs_mag = dict(title='Magnitude', label_font_size=16, title_font_size=18, vertical=True, position_y=0.1, height=0.8, color='black')
    sargs_err = dict(title='Abs. Error', label_font_size=16, title_font_size=18, vertical=True, position_y=0.1, height=0.8, color='black')

    # --- 3. Generate All Plots ---
    generate_plot(gt_mesh, "Ground Truth (Deformed)", 'magnitude', mag_clim, 'jet', sargs_mag, os.path.join(args.output_dir, '01_ground_truth.png'), warp_factor=args.warp_factor)
    
    for i, name in enumerate(args.names):
        mesh = all_meshes[name]
        if mesh.n_points > 0:
            generate_plot(mesh, f"{name} Prediction (Deformed)", 'magnitude', mag_clim, 'jet', sargs_mag, os.path.join(args.output_dir, f'{i+2:02d}_{name.replace(" ","_")}_prediction.png'), warp_factor=args.warp_factor)
            generate_plot(mesh, f"{name} Error", 'error', error_clim, 'Reds', sargs_err, os.path.join(args.output_dir, f'{i+2:02d}_{name.replace(" ","_")}_error.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate high-quality, individual plots for paper figures.")
    parser.add_argument('--files', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--warp_factor', type=float, default=50.0, help="Factor to exaggerate deformation.")
    args = parser.parse_args()
    main(args)