import torch
import numpy as np

def tetra_to_edges(topology):
    """
    Converts a tensor of tetrahedral elements to a graph edge index.
    A tetrahedron (i, j, k, l) has 6 edges. A tetrahedron (i, j, k, l) has 6 edges: (i,j), (i,k), (i,l), (j,k), (j,l), (k,l).
    This function finds all unique edges and returns them in a format suitable for PyG (undirected).

    Args:
        topology (np.array or torch.Tensor): Array of shape (num_tetra, 4)
                                             containing node indices for each tetrahedron.

    Returns:
        torch.Tensor: A tensor of shape (2, num_edges) representing the
                      undirected graph connectivity (edge_index).
    """
    # Extract unique edges from the tetrahedral elements
    topology = np.asarray(topology)
    edges = set()
    for tetra in topology:
        # Sort to ensure (i, j) and (j, i) are treated as the same edge
        edges.add(tuple(sorted((tetra[0], tetra[1]))))
        edges.add(tuple(sorted((tetra[0], tetra[2]))))
        edges.add(tuple(sorted((tetra[0], tetra[3]))))
        edges.add(tuple(sorted((tetra[1], tetra[2]))))
        edges.add(tuple(sorted((tetra[1], tetra[3]))))
        edges.add(tuple(sorted((tetra[2], tetra[3]))))

    # Convert the set of edges to a torch tensor
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # PyG expects an undirected graph, so we add the reverse edges
    # The set-based creation already handles uniqueness, so we just need to create both directions   
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)