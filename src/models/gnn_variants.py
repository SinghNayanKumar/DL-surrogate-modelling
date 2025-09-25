import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MetaLayer
from torch_scatter import scatter_add

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
    """
    The EdgeModel computes a "message" for each edge.
    It takes the features of the source and destination nodes of an edge
    and returns a new feature vector (the message) for that edge.
    """
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
    """
    The NodeModel updates the features of each node.
    It aggregates all incoming messages for a node and combines them
    with the node's current features to produce an updated feature vector.
    """

    def __init__(self, hidden_size):
        super(NodeModel, self).__init__()
        # The input to the MLP is the node's own features concatenated
        # with the aggregated features from its neighbors' messages.
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x has shape [num_nodes, hidden_size]
        # edge_index has shape [2, num_edges]
        # edge_attr (messages from EdgeModel) has shape [num_edges, hidden_size]
        
        row, col = edge_index # row = source nodes, col = destination nodes
        
        # We use scatter_add to sum up all messages (`edge_attr`) that are
        # destined for the same node. The `col` tensor tells scatter_add
        # which messages belong to which destination node.
        # The output `aggregated_messages` will have shape [num_nodes, hidden_size].
        aggregated_messages = scatter_add(edge_attr, col, dim=0, dim_size=x.size(0))
        
        # Now, we combine the node's original features with the aggregated messages.
        # This is the core of message passing.
        combined_features = torch.cat([x, aggregated_messages], dim=1)
        
        # The MLP processes this combined feature vector to get the final updated node embedding.
        # The output has the correct shape: [num_nodes, hidden_size].
        return self.mlp(combined_features)

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
        
        # MetaLayer internally calls EdgeModel first, then NodeModel.
        for layer in self.processor:
            # The edge_model receives (x[row], x[col], ...), which is correct.
            # The node_model will now correctly return a tensor of shape [num_nodes, hidden_size].
            x_updated, _, _ = layer(x, edge_index)
            # The residual connection will now work because x and x_updated have the same shape.
            x = x + x_updated 
        
        x = self.decoder(x)
        return x