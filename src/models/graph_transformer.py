import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from src.models.gnn_variants import GNN_Base

class GraphTransformer_Surrogate(GNN_Base):
    """ 
    GNN using Graph Transformer layers. 
    Inherits conditional activation logic from GNN_Base.
    """
    def __init__(self, **kwargs):
        # This will call GNN_Base.__init__ and set up self.activation
        super(GraphTransformer_Surrogate, self).__init__(**kwargs)
        hidden_size = kwargs.get('hidden_size', 128)
        num_heads = kwargs.get('num_heads', 4)
        self.processor = nn.ModuleList()
        
        # TransformerConv is a powerful layer using self-attention on node neighborhoods
        self.processor.append(TransformerConv(hidden_size, hidden_size, heads=num_heads))
        # Note: The input to the next layer is hidden_size * num_heads
        self.processor.append(TransformerConv(hidden_size * num_heads, hidden_size, heads=num_heads))
        self.processor.append(TransformerConv(hidden_size * num_heads, hidden_size, heads=1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.encoder(x)
        for layer in self.processor:
            # Use the activation function defined in the base class
            x = self.activation(layer(x, edge_index))
        x = self.decoder(x)
        return x