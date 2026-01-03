import torch

from torch import nn
from .utils.wrapper import DtypeAlignWrapper
from torch_geometric.nn import TransformerConv


class TMSEmbedding(nn.Module): 
    """
    Traffic Matrix Sequence Embedding module that processes traffic matrix sequences.
    Uses a linear layer followed by an LSTM to embed traffic matrix sequences.
    """

    def __init__(self, num_nodes, d_middle, d_model, num_rnn_layers=1): 
        """
        Initialize the TMSEmbedding module.
        
        Args:
            num_nodes: Number of nodes in the network
            d_middle: Dimension of the intermediate linear layer
            d_model: Hidden size of the LSTM
            num_rnn_layers: Number of LSTM layers
        """
        super().__init__()
        # Linear layer to project traffic matrix to intermediate dimension
        self.linear = DtypeAlignWrapper(nn.Linear(num_nodes * (num_nodes - 1), d_middle))
        # LSTM to process the sequence of traffic matrices
        self.lstm = DtypeAlignWrapper(nn.LSTM(
            input_size=d_middle, 
            hidden_size=d_model, 
            num_layers=num_rnn_layers, 
            batch_first=False,  # Expected input format: (seq_len, batch, features)
            bidirectional=False  # Unidirectional LSTM
        ))
        self.act = nn.ReLU()  # Activation function after linear layer
        
    def forward(self, x):
        """
        Forward pass of the TMSEmbedding module.
        
        Args:
            x: Input traffic matrix sequences of shape [batch_size, seq_len, num_nodes*(num_nodes-1)]
            
        Returns:
            Embedded traffic matrix sequences of shape [batch_size, seq_len, d_model]
        """
        self.lstm.flatten_parameters()  # Optimize LSTM parameters for performance
        x = self.act(self.linear(x))  # Apply linear projection and activation
        x = x.transpose(0, 1).contiguous()  # Transpose to [seq_len, batch_size, d_middle] for LSTM
        output, *_ = self.lstm(x)  # Process through LSTM
        return output.transpose(0, 1).contiguous()  # Transpose back to [batch_size, seq_len, d_model]


class GraphProcessing(nn.Module): 
    """
    Graph processing module using TransformerConv layers to process graph-structured data.
    Applies multiple layers of graph transformers to learn node representations.
    """

    def __init__(
        self, 
        num_features, 
        num_layers, 
        num_heads=4,
        hidden_dim=32,
        dropout=0.1
    ):
        """
        Initialize the GraphProcessing module.
        
        Args:
            num_features: Number of input features per node
            num_layers: Number of TransformerConv layers
            num_heads: Number of attention heads in each TransformerConv layer
            hidden_dim: Hidden dimension of each TransformerConv layer
            dropout: Dropout rate applied after each layer
        """
        super(GraphProcessing, self).__init__()
        self.layers = nn.ModuleList()  # List to store the TransformerConv layers
        self.dropout = nn.Dropout(dropout)  # Dropout layer applied after each TransformerConv layer
        
        for i in range(num_layers):
            # Create TransformerConv layer with appropriate input dimensions
            self.layers.append(
                DtypeAlignWrapper(TransformerConv(
                    in_channels=num_features if i == 0 else hidden_dim,  # Input dim: num_features for first layer, hidden_dim for others
                    out_channels=hidden_dim,  # Output dimension of the layer
                    heads=num_heads,  # Number of attention heads
                    concat=False,  # Don't concatenate head outputs (average them instead)
                    dropout=dropout,  # Attention dropout
                    edge_dim=None,  # No edge features used in this implementation
                ))
            )

    def forward(self, x, edge_index, capacities=None):
        """
        Forward pass of the GraphProcessing module.
        
        Args:
            x: Node features of shape [num_nodes, num_features]
            edge_index: Graph connectivity in COO format [2, num_edges]
            capacities: Edge capacities (optional)
            
        Returns:
            Edge embeddings that combine node representations and edge capacities
        """
        # Process through each TransformerConv layer
        for layer in self.layers:
            x = layer(x, edge_index)  # Apply graph transformer convolution
            x = torch.relu(x)  # Apply ReLU activation
            x = self.dropout(x)  # Apply dropout
        
        # Create edge embeddings by combining node representations
        # For each edge, sum the representations of its source and target nodes
        edge_embeddings = x[edge_index.transpose(0, 1).contiguous()].sum(dim=1)
        
        # Concatenate edge embeddings with capacity information if provided
        if capacities is not None:
            edge_embeddings = torch.cat((edge_embeddings, capacities.unsqueeze(-1).contiguous()), dim=-1)
        
        return edge_embeddings
