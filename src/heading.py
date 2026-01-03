import math
import torch

from torch import nn
from .utils.wrapper import DtypeAlignWrapper


class SinusoidalPosEmb(nn.Module):
    """
    Implements sinusoidal positional embedding, which is commonly used in transformer models.
    This embedding provides information about the position of tokens in a sequence.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Generates sinusoidal positional embeddings based on input positions.
        
        Args:
            x: Input positions (typically indices)
            
        Returns:
            Sinusoidal positional embeddings with shape (batch_size, embedding_dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]  # (batch_size, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # Concatenate sin and cos parts
        return emb


class IndexFusion(nn.Module): 
    """
    Fuses input features with index-based positional embeddings.
    This module applies sinusoidal positional embeddings to input features based on an index.
    """

    def __init__(self, hidden_dimmension, dropout=0.): 
        super().__init__()
        self.act = nn.ReLU()
        # Create sinusoidal positional embeddings
        self.emb = SinusoidalPosEmb(hidden_dimmension)

        # Linear projection to expand dimensions for scale and shift
        self.linear_project = DtypeAlignWrapper(nn.Linear(hidden_dimmension, hidden_dimmension * 2))

    def forward(self, x, index=None, index_range=None):
        """
        Forward pass that fuses input features with positional embeddings.
        
        Args:
            x: Input tensor
            index: Specific index to use for embedding (optional)
            index_range: Range of indices to generate embeddings for
            
        Returns:
            Tensor with positional information fused in
        """
        if index is None:
            # Generate embeddings for a range of indices
            index_instance = torch.arange(index_range, device=x.device, dtype=torch.long).unsqueeze(0)
            index_tensor = index_instance.repeat(x.shape[0], 1).reshape(-1)
            emb = self.emb(index_tensor)
            emb = self.linear_project(self.act(emb)).reshape(x.shape[0], index_range, -1)
            scale, shift = torch.chunk(emb, 2, dim=-1)  # Split into scale and shift components
            x = x.unsqueeze(1) * (1 + scale) + shift  # Apply scale and shift transformation
        else:
            # Generate embedding for a specific index
            index_tensor = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
            emb = self.emb(index_tensor)
            emb = self.linear_project(self.act(emb))
            scale, shift = torch.chunk(emb, 2, dim=1)  # Split into scale and shift components
            x = x.unsqueeze(1) * (1 + scale) + shift  # Apply scale and shift transformation
        
        return x
    

class LightWeightHead(nn.Module): 
    
    def __init__( 
        self, 
        in_dim, 
        hidden_dim, 
        num_nodes,
        num_paths,
        num_layers=4,
        dropout=0.
    ): 
        """
        A lightweight output head for traffic engineering models that predicts path split ratios.
        Uses IndexFusion to incorporate node pair information in the prediction process.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden layer dimension
            num_nodes: Number of nodes in the network
            num_paths: Number of paths per OD pair
            num_layers: Number of hidden layers in the MLP
            dropout: Dropout rate
        """
        super().__init__() 

        self.num_nodes, self.num_paths = num_nodes, num_paths 

        assert num_layers >= 1, "num_layers must be at least 1"

        # Input projection layer to map input to hidden dimension
        self.in_proj = DtypeAlignWrapper(nn.Linear(in_dim, hidden_dim))
        # Learnable temperature parameter for scaling the output
        self.temperature = nn.Parameter(torch.zeros(1, num_nodes, 1))

        # Build the hidden layers of the MLP
        layers = []
        layers.append(DtypeAlignWrapper(nn.Linear(hidden_dim, hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(DtypeAlignWrapper(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        if num_layers > 1:
            layers.append(DtypeAlignWrapper(nn.Linear(hidden_dim, (num_nodes-1) * num_paths)))
        else:
            layers[0] = DtypeAlignWrapper(nn.Linear(hidden_dim, (num_nodes-1) * num_paths))

        self.out_proj = nn.Sequential(*layers)

        # IndexFusion module to incorporate node pair information
        self.od_emb = IndexFusion(hidden_dim, dropout=dropout) 

    def forward(self, input_embedding, pair_id=None): 
        """
        Forward pass of the LightWeightHead.
        
        Args:
            input_embedding: Input embeddings to the head
            pair_id: Optional specific OD pair ID
            
        Returns:
            Softmax-normalized split ratios for paths in the network
        """
        B, *_ = input_embedding.shape 
        # Apply OD pair embedding to the input
        x = self.od_emb(self.in_proj(input_embedding), index_range=self.num_nodes, index=pair_id) 
        # Apply output projection and reshape to separate nodes and paths
        output = self.out_proj(x).reshape(B, -1, self.num_paths).contiguous() 
        # Apply temperature scaling
        output = output * self.temperature.exp().repeat(B, self.num_nodes-1, self.num_paths).contiguous() 
        # Apply softmax to get split ratios
        split_ratios = torch.nn.functional.softmax(output, dim=-1) 
        return split_ratios 
    

class SoftMaxHead(nn.Module): 

    def __init__( 
        self, 
        in_dim, 
        hidden_dim, 
        output_dim,  # The original code had output_dim instead of num_nodes and num_paths
        dropout=0.
    ): 
        """
        A softmax output head for traffic engineering models that predicts path split ratios.
        This is a simpler alternative to the LightWeightHead.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (num_nodes * (num_nodes-1) * num_paths)
            dropout: Dropout rate
        """
        super().__init__() 

        self.output_projection = nn.Sequential(
            # nn.Flatten(start_dim = -2),  # (b, s, d) -> (b * s, d)
            # nn.Linear((seq_len + 0) * d_model, d_middle), 
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)  # Output for all OD pairs and paths
        )

    def forward(self, input_embedding):
        """
        Forward pass of the SoftMaxHead.
        
        Args:
            input_embedding: Input embeddings to the head
            
        Returns:
            Softmax-normalized split ratios for paths in the network
        """
        # Apply the output projection network
        split_ratios = self.output_projection(input_embedding)
        
        # Note: The original code had a reference to self.num_paths which is not set in __init__
        # For the code to work properly, we need to calculate the number of paths
        # from the context. In a real implementation, this would be passed as a parameter
        # or calculated based on network parameters.
        # For now, we'll need to determine the number of paths from the context
        # or from the model configuration.
        
        # Reshape and apply softmax to get valid probability distributions over paths
        # The number of paths needs to be determined based on the expected output structure
        # This implementation assumes that the last dimension after reshaping is the number of paths
        split_ratios = torch.nn.functional.softmax(split_ratios.reshape(split_ratios.shape[0], -1, self.output_dim // (split_ratios.shape[0] * (split_ratios.shape[1] // split_ratios.shape[0]))), dim=-1)
        
        # Flatten the output for further processing
        split_ratios = split_ratios.reshape(split_ratios.shape[0], -1)
        return split_ratios