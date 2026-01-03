import math
import torch

from torch import nn


class ReprogrammingLayer(nn.Module):
    """
    Implements a reprogramming layer that adapts embeddings from one space to another using attention mechanism.
    
    This layer is designed to reprogram embeddings from a source domain (e.g., LLM embeddings) 
    to be compatible with a target domain (e.g., traffic embeddings) using a multi-head attention mechanism.
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        """
        Initialize the ReprogrammingLayer.
        
        Args:
            d_model: Dimension of the target embeddings to be reprogrammed
            n_heads: Number of attention heads
            d_keys: Dimension of the keys and queries (default: d_model // n_heads)
            d_llm: Dimension of the source embeddings (LLM embeddings)
            attention_dropout: Dropout rate for attention weights
        """
        super(ReprogrammingLayer, self).__init__()

        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)

        # Linear projections for query, key, and value
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # Projects target embeddings to query space
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)      # Projects source embeddings to key space
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)    # Projects source embeddings to value space
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)      # Projects output back to LLM dimension
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Forward pass of the reprogramming layer.
        
        Args:
            target_embedding: Target embeddings to be reprogrammed, shape [B, L, d_model]
            source_embedding: Source embeddings to adapt from (keys), shape [S, d_llm]
            value_embedding: Source embeddings to adapt from (values), shape [S, d_llm]
            
        Returns:
            Reprogrammed embeddings in the target space, shape [B, L, d_llm]
        """
        # Extract dimensions
        B, L, _ = target_embedding.shape  # Batch size, sequence length of target
        S, _ = source_embedding.shape     # Sequence length of source
        H = self.n_heads                # Number of attention heads

        # Apply linear projections and reshape for multi-head attention
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)  # [B, L, H, d_keys]
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)        # [S, H, d_keys]
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)        # [S, H, d_keys]

        # Calculate attention scores using scaled dot-product attention
        scale = 1. / math.sqrt(target_embedding.shape[-1])  # Scale factor for attention
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)  # [B, H, L, S]
        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # Apply softmax and dropout
        out = torch.einsum("bhls,she->blhe", A, value_embedding)  # [B, L, H, d_keys]
        
        # Reshape and apply output projection
        out = out.reshape(B, L, -1)  # [B, L, H * d_keys]
        return self.out_projection(out)  # [B, L, d_llm]
