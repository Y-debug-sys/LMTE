import torch
import random
import numpy as np


def set_seed(seed, cudnn=False):
    """
    Set random seeds for reproducible results across multiple libraries.
    
    Args:
        seed (int): Random seed value to use for all random number generators
        cudnn (bool): Whether to set CUDA-specific options for reproducibility
    """
    # Set seed for NumPy random number generator
    np.random.seed(seed)
    
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for PyTorch's CPU random number generator
    torch.manual_seed(seed)

    # If CUDA is available, set seeds for GPU random number generators
    if torch.cuda.is_available():
        # Set seed for the current GPU
        torch.cuda.manual_seed(seed)
        # Set seed for all available GPUs (useful for multi-GPU setups)
        torch.cuda.manual_seed_all(seed)

    if cudnn:
        # Ensure deterministic behavior for convolution operations to guarantee reproducibility
        # These options may impact performance slightly but ensure consistent results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
