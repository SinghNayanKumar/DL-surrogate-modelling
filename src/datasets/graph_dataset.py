# graph_dataset.py

import os
import glob
import torch
import h5py
from torch_geometric.data import Dataset, Data
from src.utils.mesh_utils import tetra_to_edges

class IBeamGraphDataset(Dataset):
    """
    Custom PyG Dataset for loading I-Beam FEA simulation data from H5 files.
    This dataset can be configured to generate node features with or without one-hot encoding
    for the 'load_type' parameter, making it suitable for both multi-modal (generalist)
    and unimodal (specialist) training regimes.
    """

    def __init__(self, root_dir, h5_file_list, stats=None, use_one_hot=False):
        """
        Args:
            root_dir (string): Directory where processed data might be stored.
            h5_file_list (list): List of specific H5 files to include.
            stats (dict): Dictionary with normalization constants for inputs (x) and outputs (y).
            use_one_hot (bool): If True, encodes 'load_type' as a one-hot vector (for multi-modal).
                                If False, encodes 'load_type' as a single scalar (for unimodal).
        """
        self.root_dir = root_dir
        self.h5_files = h5_file_list
        # ### --- CHANGE --- ### Added the use_one_hot flag
        self.use_one_hot = use_one_hot
        
        if stats:
            self.mean_y = stats.get('mean_y')
            self.std_y = stats.get('std_y')
            self.mean_x = stats.get('mean_x')
            self.std_x = stats.get('std_x')
        else:
            self.mean_y, self.std_y, self.mean_x, self.std_x = None, None, None, None
        super(IBeamGraphDataset, self).__init__(root_dir)

    def len(self):
        return len(self.h5_files)

    def get(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, 'r') as f:
            node_coords = torch.from_numpy(f['node_coordinates'][:]).to(torch.float32)
            topology = f['topology'][:]
            edge_index = tetra_to_edges(topology)
            displacement = torch.from_numpy(f['displacement'][:]).to(torch.float32)
            
            if self.mean_y is not None and self.std_y is not None:
                normalized_displacement = (displacement - self.mean_y) / self.std_y
            #else:
                #normalized_displacement = displacement
            
            # --- 4. Node Features (x): This is the actual input to the GNN layers. ---
            load_type_str = f.attrs.get('load_type', 'bending_y')
            
            # ### --- CHANGE START --- ###
            # Conditionally build the feature vector based on the use_one_hot flag.
            
            if self.use_one_hot:
                # --- GENERALIST (Multi-modal) PATH ---
                one_hot_map = {"bending_y": [1.0, 0.0, 0.0], "bending_x": [0.0, 1.0, 0.0], "torsion":   [0.0, 0.0, 1.0]}
                one_hot_load_type = torch.tensor(one_hot_map.get(load_type_str, [1.0, 0.0, 0.0]), dtype=torch.float32)

                params_list = [
                    f.attrs.get('beam_length', 300.0), f.attrs.get('flange_width', 100.0),
                    f.attrs.get('flange_thickness', 15.0), f.attrs.get('web_thickness', 10.0),
                    f.attrs.get('beam_depth', 150.0), f.attrs.get('fillet_radius', 12.0),
                    f.attrs.get('youngs_modulus', 2.1e11), f.attrs.get('poissons_ratio', 0.3),
                    f.attrs.get('force_magnitude', 1500.0),
                    {"uniform": 0.0, "linear_y": 1.0}.get(f.attrs.get('load_distribution', 'uniform'), 0.0)
                ]
                params = torch.tensor(params_list, dtype=torch.float32)

                num_nodes = node_coords.shape[0]
                params_per_node = params.unsqueeze(0).repeat(num_nodes, 1)
                one_hot_per_node = one_hot_load_type.unsqueeze(0).repeat(num_nodes, 1)
                
                # Final features: [coords(3), params(10), one_hot(3)] = 16 features
                node_features = torch.cat([node_coords, params_per_node, one_hot_per_node], dim=1)
            else:
                # --- SPECIALIST (Unimodal) PATH ---
                load_type_map = {"bending_y": 0.0, "bending_x": 1.0, "torsion": 2.0}
                
                params_list = [
                    f.attrs.get('beam_length', 300.0), f.attrs.get('flange_width', 100.0),
                    f.attrs.get('flange_thickness', 15.0), f.attrs.get('web_thickness', 10.0),
                    f.attrs.get('beam_depth', 150.0), f.attrs.get('fillet_radius', 12.0),
                    f.attrs.get('youngs_modulus', 2.1e11), f.attrs.get('poissons_ratio', 0.3),
                    f.attrs.get('force_magnitude', 1500.0),
                    load_type_map.get(load_type_str, 0.0), # Use scalar load type
                    {"uniform": 0.0, "linear_y": 1.0}.get(f.attrs.get('load_distribution', 'uniform'), 0.0)
                ]
                params = torch.tensor(params_list, dtype=torch.float32)

                num_nodes = node_coords.shape[0]
                params_per_node = params.unsqueeze(0).repeat(num_nodes, 1)

                # Final features: [coords(3), params(11)] = 14 features
                node_features = torch.cat([node_coords, params_per_node], dim=1)
            
            # ### --- CHANGE END --- ###
            
            if self.mean_x is not None and self.std_x is not None:
                normalized_features = (node_features - self.mean_x) / self.std_x
            else:
                normalized_features = node_features

            material_props = torch.tensor([[f.attrs.get('youngs_modulus', 2.1e11), f.attrs.get('poissons_ratio', 0.3)]], dtype=torch.float32)

            graph_data = Data(
                x=normalized_features, 
                edge_index=edge_index,
                y=normalized_displacement,
                pos=node_coords,
                material_props=material_props
            )
            
        return graph_data