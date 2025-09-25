import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MetaLayer

class GNN_Base(nn.Module):
    """ Base class for GNNs to share encoder/decoder structure. """
    def __init__(self, node_in_features, node_out_features, hidden_size=128):
        super(GNN_Base, self).__init__()
        self.encoder = nn.Linear(node_in_features, hidden_size)
        self.decoder = nn.Linear(hidden_size, node_out_features)

    def forward(self, data):
        raise NotImplementedError

class GCN_Surrogate(GNN_Base):
    """ GNN using Graph Convolutional Network (GCN) layers. """
    def __init__(self, **kwargs):
        super(GCN_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        self.processor = nn.ModuleList()
        for _ in range(3): # 3 layers of message passing
            self.processor.append(GCNConv(hidden_size, hidden_size))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        for layer in self.processor:
            x = torch.relu(layer(x, edge_index))
        x = self.decoder(x)
        return x

class GAT_Surrogate(GNN_Base):
    """ GNN using Graph Attention Network (GAT) layers. """
    def __init__(self, **kwargs):
        super(GAT_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        num_heads = kwargs.get('num_heads', 4)
        self.processor = nn.ModuleList()
        # Input layer
        self.processor.append(GATConv(hidden_size, hidden_size, heads=num_heads, concat=True))
        # Hidden layer
        self.processor.append(GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=True))
        # Output layer
        self.processor.append(GATConv(hidden_size * num_heads, hidden_size, heads=1, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        x = torch.relu(self.processor[0](x, edge_index))
        x = torch.relu(self.processor[1](x, edge_index))
        x = self.processor[2](x, edge_index) # No activation on final layer
        x = self.decoder(x)
        return x

class EdgeModel(nn.Module):
    def __init__(self, hidden_size):
        super(EdgeModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], dim=1)
        return self.mlp(out)

class NodeModel(nn.Module):
    def __init__(self, hidden_size):
        super(NodeModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        return self.mlp(out)

class MPNN_Surrogate(GNN_Base):
    """ GNN using a general Message Passing Neural Network (MPNN) structure with MetaLayers. """
    def __init__(self, **kwargs):
        super(MPNN_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        self.processor = nn.ModuleList()
        for _ in range(3): # 3 layers of message passing
            self.processor.append(MetaLayer(EdgeModel(hidden_size), NodeModel(hidden_size)))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        # In a more complex model, you would also encode edge_attr and u (globals)
        for layer in self.processor:
            x_updated, _, _ = layer(x, edge_index)
            x = x + x_updated # Residual connection
        x = self.decoder(x)
        return x