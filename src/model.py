import torch

from torch import nn 
from .prompting import prepare_prompts
from .utils.wrapper import DtypeAlignWrapper
from .reprogramming import ReprogrammingLayer
from .heading import LightWeightHead, SoftMaxHead 
from .embedding import TMSEmbedding, GraphProcessing

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, AutoTokenizer 


class LmteModel(nn.Module): 
    """
    LMTE (Large Model for Traffic Engineering) Model
    
    This model combines traffic matrix sequences, network topology information, and large language models
    to predict traffic split ratios for network routing optimization. It uses a hybrid approach that
    integrates graph neural networks, temporal modeling, and LLM-based reasoning.
    
    The architecture follows these main steps:
    1. Process network topology using GNNs to generate structural embeddings
    2. Process traffic time series using RNN-based embedding
    3. Combine topology and traffic information with textual prompts
    4. Use a reprogramming layer to adapt the combined features to the LLM's input space
    5. Feed the processed inputs to a frozen LLM to leverage its reasoning capabilities
    6. Extract the LLM's output and project it to the desired split ratio predictions
    """

    def __init__(
        self, 
        d_model, 
        d_llm, 
        num_nodes, 
        num_paths, 
        num_eages, 
        num_gnn_layers=2, 
        num_rnn_layers=1,
        num_dense_layers=4,
        d_keys=None, 
        seq_len=12, 
        n_heads=4, 
        dropout=0.1, 
        d_middle=32, 
        llm_model='llama-8b', 
        llm_layers=1, 
        max_length=128,
        use_divide_head=False
    ):
        """
        Initialize the LMTE model with specified parameters.
        
        Args:
            d_model: Dimension of the model's internal representations
            d_llm: Dimension of the LLM's hidden states
            num_nodes: Number of nodes in the network topology
            num_paths: Number of paths between node pairs
            num_eages: Number of edges in the network (note: likely a typo, should be num_edges)
            num_gnn_layers: Number of GNN layers for graph processing
            num_rnn_layers: Number of RNN layers for temporal processing
            num_dense_layers: Number of dense layers in the output head
            d_keys: Dimension for attention keys
            seq_len: Length of the traffic matrix sequence
            n_heads: Number of attention heads
            dropout: Dropout rate for regularization
            d_middle: Middle dimension for various transformations
            llm_model: Type of LLM to use ('llama-8b', 'llama-1b', 'llama-3b', or default)
            llm_layers: Number of LLM layers to use
            max_length: Maximum length for various sequences
            use_divide_head: Whether to use lightweight head instead of standard softmax head
        """
        super(LmteModel, self).__init__()

        # Store key network dimensions
        self.num_edges, self.seq_len = num_eages, seq_len
        self.num_paths, self.num_nodes = num_paths, num_nodes
        # Calculate total output dimension: for each path and each OD pair (excluding self-pairs)
        self.output_dim = num_paths * num_nodes * (num_nodes - 1)

        # Define projection layers for tunnel embeddings
        # fc_out1: projects flattened tunnel embeddings to model dimension
        self.fc_out1 = DtypeAlignWrapper(nn.Linear((d_keys + 1) * max_length, d_model))
        # fc_out2: adjusts dimensions based on number of node pairs
        self.fc_out2 = DtypeAlignWrapper(nn.Linear(num_nodes * (num_nodes - 1), d_model))

        # Initialize specialized embedding modules
        # tms_embedding: processes traffic matrix sequences using RNN architecture
        self.tms_embedding = TMSEmbedding(num_nodes, d_middle, d_model, num_rnn_layers)
        # graph_emmbedding: processes graph structure using GNN layers with multi-head attention
        self.graph_emmbedding = GraphProcessing(2, num_gnn_layers, num_heads=n_heads, dropout=dropout, 
                                                hidden_dim=d_keys)
        # reprogramming_layer: adapts domain-specific embeddings to the LLM's input space
        self.reprogramming_layer = ReprogrammingLayer(num_paths * d_model, n_heads, d_keys, d_llm, dropout)

        """ Load and configure the Large Language Model and tokenizer. """

        # Determine model ID and directories based on the selected LLM variant
        if llm_model == 'llama-8b':
            model_id = 'meta-llama/Meta-Llama-3-8B'  # Model ID for LLaMA 3 8B model 
            model_dir = './lms/Meta-Llama-3-8B'  # Directory to save the model locally
            tokenizer_dir = './lms/Meta-Llama-3-8B/tokenizer.json'  # Directory to save the tokenizer locally
        elif llm_model == 'llama-1b':
            model_id = 'LLM-Research/Llama-3.2-1B'  # Model ID for LLaMA 3.2 1B model 
            model_dir = './lms/Llama-3.2-1B'  # Directory to save the model locally
            tokenizer_dir = './lms/Llama-3.2-1B/tokenizer.json'  # Directory to save the tokenizer locally
        elif llm_model == 'llama-3b':
            model_id = 'LLM-Research/Llama-3.2-3B'  # Model ID for LLaMA 3.2 3B model
            model_dir = './lms/Llama-3.2-3B'  # Directory to save the model locally
            tokenizer_dir = './lms/Llama-3.2-3B/tokenizer.json'  # Directory to save the tokenizer locally
        else:  # Default case for llama-13b
            model_id = 'meta-llama/Meta-Llama-2-13B'  # Model ID for LLaMA 2 13B model 
            model_dir = './lms/Llama-2-13B'  # Directory to save the model locally 
            tokenizer_dir = './lms/Llama-2-13B/tokenizer.model'  # Directory to save the tokenizer locally 

        # Configure the Llama model with custom parameters
        self.llama_config = LlamaConfig.from_pretrained(model_dir)
        self.llama_config.num_hidden_layers = llm_layers  # Limit the number of active layers
        self.llama_config.output_attentions = True  # Enable returning attention weights
        self.llama_config.output_hidden_states = True  # Enable returning hidden states

        # Try loading the model from local directory first, then from HuggingFace if not found
        try:
            self.llm_model = LlamaModel.from_pretrained(
                model_dir,
                config=self.llama_config,
                # load_in_4bit=True  # Option to load in 4-bit precision for memory efficiency
            )
        except EnvironmentError:  # Download model from HuggingFace if local files not found
            print("Local model files not found. Attempting to download...")
            self.llm_model = LlamaModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False,
                config=self.llama_config,
                # load_in_4bit=True
            )
            
        # Load tokenizer with the same fallback approach
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                local_files_only=True
            )
        except EnvironmentError:  # Download tokenizer from HuggingFace if local files not found
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=False
            )
        
        """ Configure padding token for proper batching. """
        
        # Set padding token - use EOS token if available, otherwise add a new PAD token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'  # Standard padding token format
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        # Freeze LLM parameters to prevent updates during training
        for param in self.llm_model.parameters():
            param.requires_grad = False  # Keep LLM frozen

        # Store word embeddings and prevent them from being trained
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.word_embeddings.requires_grad = False 
        self.vocab_size = self.word_embeddings.shape[0]
        # Mapping layer to reduce embedding dimensionality for reprogramming
        self.mapping_layer = nn.Linear(self.vocab_size, 1000)

        # Define flatten operation for reshaping tensors before final projection
        self.flatten = nn.Flatten(start_dim=-2)  # (b, s, d) -> (b * s, d)

        self.d_model = d_model
        # Calculate the total number of patches (temporal sequence + network edges)
        self.patch_nums = seq_len + num_eages

        # Determine output dimension for LLM based on total input size
        # Ensure compatibility between our feature dimension and LLM's expected input
        if (seq_len + num_eages) * d_model > d_llm:
            self.llm_out_dim = d_llm
        else:
            self.llm_out_dim = (seq_len + num_eages) * d_model

        # Initialize the output projection head based on configuration
        # LightWeightHead is more efficient, SoftMaxHead may provide better accuracy
        if use_divide_head:
            self.softmax_proj = LightWeightHead(self.llm_out_dim, d_middle, num_nodes, num_paths, 
                                                num_dense_layers, dropout)
        else:
            self.softmax_proj = SoftMaxHead(self.llm_out_dim, d_middle, self.output_dim, dropout)

    def forward(
        self, 
        tms,  # Traffic matrix sequence [B, seq_len, num_nodes, num_nodes]
        node_features,  # Node features [num_nodes, feature_dim]
        edge_index,  # Edge connections [2, num_edges]
        capacities,  # Edge capacities [num_edges]
        topology,  # Network topology information
        burst_situations=None,  # Optional burst situation descriptions
        padded_edge_ids_per_path=None  # Padded edge IDs for each path [num_paths, max_path_length]
    ):
        """
        Forward pass of the LMTE model.
        
        Processes network state and traffic data to predict traffic split ratios.
        
        Args:
            tms: Traffic matrix sequence
            node_features: Features of nodes in the graph
            edge_index: Edge connections in the graph
            capacities: Capacities of edges in the network
            topology: Network topology information
            burst_situations: Optional description of current burst situations
            padded_edge_ids_per_path: Padded edge IDs for each path
            
        Returns:
            split_ratios: Predicted traffic split ratios [B, output_dim]
        """
        # Get batch size and number of nodes from input tensors
        B, N = tms.shape[0], node_features.shape[0]

        # Prepare textual prompts that describe the network state
        # This combines topology information with current traffic conditions
        prompt = prepare_prompts(topology, tms, self.num_paths, burst_situations)
        
        # Tokenize the prompt for LLM input
        # Converts text prompts to token IDs with proper padding and truncation
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids

        # Convert token IDs to embeddings using the LLM's embedding layer
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(tms.device))  # (batch, prompt_token, dim)
        
        # Transform word embeddings using the mapping layer for reprogramming
        text_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0).contiguous()).permute(1, 0).contiguous()
        
        # Generate topology embeddings using graph processing module
        # Captures structural information of the network
        topo_embedding = self.graph_emmbedding(node_features, edge_index, capacities)
        
        # Extract tunnel embeddings for each path using padded edge IDs
        tunnel_embedding = topo_embedding[padded_edge_ids_per_path]
        
        # Create mask for padding indices and set their embeddings to zero
        padding_index = torch.where(padded_edge_ids_per_path.unsqueeze(-1).repeat(1, 1, tunnel_embedding.shape[-1]) == -1)
        tunnel_embedding[padding_index] = 0.
        
        # Reshape tunnel embeddings to the required format
        tunnel_embedding = tunnel_embedding.reshape(tunnel_embedding.shape[0], -1).contiguous()

        # Process tunnel embeddings through output projection layers
        topo_embedding = self.fc_out1(tunnel_embedding.contiguous())
        topo_embedding = topo_embedding.reshape(-1, self.num_paths * topo_embedding.shape[-1]).contiguous()
        topo_embedding = self.fc_out2(topo_embedding.transpose(0,1).contiguous())
        
        # Process traffic matrix sequences through temporal embedding module
        history_embedding = self.tms_embedding(tms.contiguous())
        
        # Combine history and topology embeddings using einsum operation
        # This creates a joint representation of temporal and structural information
        x = torch.einsum("bsd,dc->bsc", history_embedding, topo_embedding.transpose(0,1).contiguous()).contiguous()

        # Apply reprogramming layer to adapt the combined embeddings to LLM space
        # Uses text embeddings as queries and keys for cross-attention
        enc_out = self.reprogramming_layer(x, text_embeddings, text_embeddings)
        
        # Concatenate prompt embeddings with processed input embeddings
        # Creates the complete input sequence for the LLM
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1).contiguous()

        # Process the combined embeddings through the LLM
        # The LLM performs reasoning on the network state
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        
        # Extract the relevant portion of the output for final prediction
        # Flattens the output and takes the last part corresponding to our input
        projector_input = self.flatten(dec_out)[:, - self.llm_out_dim:] 
        
        # Apply the softmax projection head to get split ratios
        # Converts LLM output to the desired traffic engineering output format
        split_ratios = self.softmax_proj(projector_input) 

        # Reshape output to the required format [B, output_dim]
        split_ratios = split_ratios.reshape(B, -1) 

        # Return split ratios with proper type and contiguous memory layout
        return split_ratios.float().contiguous() 
