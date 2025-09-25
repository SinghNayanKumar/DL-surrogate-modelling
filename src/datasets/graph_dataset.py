import os
import glob
import torch
import h5py
from torch_geometric.data import Dataset, Data
from src.utils.mesh_utils import tetra_to_edges

class IBeamGraphDataset(Dataset):
    """
    Custom PyG Dataset for loading I-Beam FEA simulation data from H5 files.
    Each sample in the dataset is a single graph representing one FEA simulation.
    This dataset is responsible for constructing a graph with rich node features
    that include both positional information and the physical parameters of the simulation.
    """

    def __init__(self, root_dir, h5_file_list, stats=None):
        """
        Args:
            root_dir (string): Directory where processed data might be stored (a PyG convention).
            h5_file_list (list): List of specific H5 files to include in this dataset instance (e.g., train/val/test).
            stats (dict): A dictionary containing 'mean_y' and 'std_y' tensors for target normalization.
        """
        self.root_dir = root_dir
        self.h5_files = h5_file_list
        if stats:
            self.mean_y = stats['mean_y']
            self.std_y = stats['std_y']
        else:
            self.mean_y = None
            self.std_y = None
        super(IBeamGraphDataset, self).__init__(root_dir)

    def len(self):
        """Returns the total number of samples (graphs) in the dataset."""
        return len(self.h5_files)

    def get(self, idx):
        """
        Generates one sample of data.
        This method reads a single .h5 file and converts it into a torch_geometric.data.Data object.
        This object contains all the information needed for one training step: node features,
        connectivity, and ground truth labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, 'r') as f:
            # --- 1. Node Positions (pos): The raw 3D coordinates of each mesh node. ---
            # This is kept separate and is primarily used for positional information.
            # Shape: [num_nodes, 3]
            node_coords = torch.tensor(f['node_coordinates'][:], dtype=torch.float)

            # --- 2. Graph Connectivity (edge_index): Derived from the mesh topology. ---
            # Describes which nodes are connected.
            # Shape: [2, num_edges]
            topology = f['topology'][:]
            edge_index = tetra_to_edges(topology)

            # --- 3. Ground Truth Labels (y): The 3D displacement vector for each node. ---
            # This is what our GNN will learn to predict.
            # Shape: [num_nodes, 3]
            displacement = torch.tensor(f['displacement'][:], dtype=torch.float)
            
            # Conditionally normalize the target displacement
            if self.mean_y is not None and self.std_y is not None:
                normalized_displacement = (displacement - self.mean_y) / self.std_y
            else:
                # If no stats are provided, use the raw displacement
                normalized_displacement = displacement

            
            # --- 4. Node Features (x): This is the actual input to the GNN layers. ---
            # We create rich features by combining positional info with physical parameters.
            # A good feature vector helps the model generalize.
            
             # 4a. Define a mapping for categorical variables
            load_type_map = {"bending_y": 0.0, "bending_x": 1.0, "torsion": 2.0}
            load_dist_map = {"uniform": 0.0, "linear_y": 1.0}

            # 4b. Create a tensor of all scalar parameters
            params_list = [
                f.attrs.get('beam_length', 300.0),
                f.attrs.get('flange_width', 100.0),
                f.attrs.get('flange_thickness', 15.0),
                f.attrs.get('web_thickness', 10.0),
                f.attrs.get('beam_depth', 150.0),
                f.attrs.get('fillet_radius', 12.0),
                f.attrs.get('youngs_modulus', 2.1e11),
                f.attrs.get('poissons_ratio', 0.3),
                f.attrs.get('force_magnitude', 1500.0),
                load_type_map.get(f.attrs.get('load_type', 'bending_y'), 0.0),
                load_dist_map.get(f.attrs.get('load_distribution', 'uniform'), 0.0)
            ]
            params = torch.tensor(params_list, dtype=torch.float)

            # 4c. Broadcast scalar parameters to every node.
            num_nodes = node_coords.shape[0]
            params_per_node = params.unsqueeze(0).repeat(num_nodes, 1)

            # 4d. Concatenate everything to form the final node feature vector.
            # Input: [pos_x, pos_y, pos_z, param_1, param_2, ..., param_N]
            node_features = torch.cat([node_coords, params_per_node], dim=1)

            # --- 5. Global Attributes (for PINN) ---
            # These are passed separately to the loss function
            material_props = torch.tensor([
                f.attrs.get('youngs_modulus', 2.1e11),
                f.attrs.get('poissons_ratio', 0.3)
            ], dtype=torch.float)

            # --- Construct the PyG Data object ---
            graph_data = Data(
                x=node_features,           # Node input features [N, 3 + num_params]
                edge_index=edge_index,
                y=normalized_displacement,
                pos=node_coords,
                material_props=material_props
            )
            
        return graph_data