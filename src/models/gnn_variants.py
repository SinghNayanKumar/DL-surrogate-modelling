import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MetaLayer
from torch_scatter import scatter_add

class GNN_Base(nn.Module):
    """ 
    Base class for GNNs. It handles the conditional activation function logic.
    If use_pinn is True, it uses SiLU (a smooth, twice-differentiable function).
    Otherwise, it defaults to ReLU for standard data-only training.
    """
    def __init__(self, node_in_features, node_out_features, hidden_size=128, use_pinn=False, **kwargs):
        super(GNN_Base, self).__init__()
        self.encoder = nn.Linear(node_in_features, hidden_size)
        self.decoder = nn.Linear(hidden_size, node_out_features)
        
        # --- CONDITIONAL ACTIVATION ---
        self.activation = nn.SiLU() if use_pinn else nn.ReLU()
        if use_pinn:
            print("[GNN_Base] PINN mode enabled: Using SiLU activation for second-order derivatives.")
        else:
            print("[GNN_Base] Data-only mode: Using standard ReLU activation.")

    def forward(self, data):
        raise NotImplementedError

# ... GCN_Surrogate and GAT_Surrogate are unchanged from the previous version ...
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
            x = self.activation(layer(x, edge_index))
        x = self.decoder(x)
        return x
    
class GAT_Surrogate(GNN_Base):
    """ GNN using Graph Attention Network (GAT) layers. """
    def __init__(self, **kwargs):
        super(GAT_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        num_heads = kwargs.get('num_heads', 4)
        self.processor = nn.ModuleList()
        self.processor.append(GATConv(hidden_size, hidden_size, heads=num_heads, concat=True))
        self.processor.append(GATConv(hidden_size * num_heads, hidden_size, heads=num_heads, concat=True))
        self.processor.append(GATConv(hidden_size * num_heads, hidden_size, heads=1, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        x = self.activation(self.processor[0](x, edge_index))
        x = self.activation(self.processor[1](x, edge_index))
        x = self.processor[2](x, edge_index)
        x = self.decoder(x)
        return x

class EdgeModel(nn.Module):
    def __init__(self, hidden_size, activation_fn):
        super(EdgeModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            activation_fn,
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest], dim=1)
        return self.mlp(out)

class NodeModel(nn.Module):
    def __init__(self, hidden_size, activation_fn):
        super(NodeModel, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            activation_fn,
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        aggregated_messages = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        combined_features = torch.cat([x, aggregated_messages], dim=1)
        return self.mlp(combined_features)

class MPNN_Surrogate(GNN_Base):
    """ GNN using a general Message Passing Neural Network (MPNN) structure with MetaLayers. """
    def __init__(self, **kwargs):
        super(MPNN_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        # === THE CHANGE: Store the use_pinn flag locally ===
        self.use_pinn = kwargs.get('use_pinn', False)
        
        self.processor = nn.ModuleList()
        for _ in range(3): 
            self.processor.append(MetaLayer(
                EdgeModel(hidden_size, self.activation), 
                NodeModel(hidden_size, self.activation)
            ))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = self.encoder(x)
        
        for layer in self.processor:
            x_updated, _, _ = layer(x, edge_index)
            
            x = x + x_updated 
        
        x = self.decoder(x)
        return x