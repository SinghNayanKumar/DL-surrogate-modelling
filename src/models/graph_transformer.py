import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from models.gnn_variants import GNN_Base

class GraphTransformer_Surrogate(GNN_Base):
    """ GNN using Graph Transformer layers. """
    def __init__(self, **kwargs):
        super(GraphTransformer_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        num_heads = kwargs.get('num_heads', 4)
        self.processor = nn.ModuleList()
        # TransformerConv is a powerful layer using self-attention on node neighborhoods
        self.processor.append(TransformerConv(hidden_size, hidden_size, heads=num_heads))
        self.processor.append(TransformerConv(hidden_size * num_heads, hidden_size, heads=num_heads))
        self.processor.append(TransformerConv(hidden_size * num_heads, hidden_size, heads=1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)
        for layer in self.processor:
            x = torch.relu(layer(x, edge_index))
        x = self.decoder(x)
        return x