import os
import glob
import torch
import h5py
from torch_geometric.data import Dataset, Data
from utils.mesh_utils import tetra_to_edges

class IBeamGraphDataset(Dataset):
    """ Custom PyG Dataset for loading I-Beam FEA simulation data from H5 files. 
    Each sample in the dataset is a single graph representing one FEA simulation."""
    def __init__(self, root_dir, h5_file_list):
        """
        Args:
            root_dir (string): Directory with all the .h5 files.
            ?transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.h5_files = h5_file_list
        super(IBeamGraphDataset, self).__init__(root_dir)

    def len(self):
        """Returns the total number of samples in the dataset."""
        return len(self.h5_files)

    def get(self, idx):
        """
        Generates one sample of data.
        Reads a single .h5 file and converts it to a torch_geometric.data.Data object.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h5_path = self.h5_files[idx]
        with h5py.File(h5_path, 'r') as f:
            # 1. Node Features (x): The 3D coordinates of each mesh node.
            # Shape: [num_nodes, 3]
            node_coords = torch.tensor(f['node_coordinates'][:], dtype=torch.float)
            
            
            # 2. Graph Connectivity (edge_index): Derived from the mesh topology.
            # Shape: [2, num_edges]
            topology = f['topology'][:]
            edge_index = tetra_to_edges(topology)

            # 3. Ground Truth Labels (y): The 3D displacement vector for each node.
            # This is what our GNN will learn to predict.
            # Shape: [num_nodes, 3]
            displacement = torch.tensor(f['displacement'][:], dtype=torch.float)
            
             # --- Construct the PyG Data object ---
            graph_data = Data(
                x=node_coords,          # Node features [N, 3] (initial coordinates)
                edge_index=edge_index,  # Graph connectivity [2, E]
                y=displacement,         # Ground truth labels [N, 3]
                pos=node_coords,        # Positional information for visualization or transforms
            )
            
            # Example of adding global features from H5 attributes
            # u = torch.tensor([[f.attrs['force_magnitude'], f.attrs['youngs_modulus']]], dtype=torch.float)
            # graph_data.u = u

        return graph_data