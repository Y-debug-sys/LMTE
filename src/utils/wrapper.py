import torch
from torch import nn


class DtypeAlignWrapper(nn.Module):
    """
    A wrapper module that ensures input tensors have the same dtype as the module's parameters.
    
    This wrapper is useful when working with mixed precision training or when modules need to 
    handle inputs with different data types. It automatically converts floating point input 
    tensors to match the dtype of the wrapped module's parameters.
    """
    def __init__(self, module: nn.Module):
        """
        Initialize the DtypeAlignWrapper.
        
        Args:
            module (nn.Module): The neural network module to wrap
        """
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        """
        Forward pass that ensures input tensors match the dtype of the wrapped module's parameters.
        
        This method detects the dtype of the wrapped module's parameters and converts
        all floating point input tensors to match that dtype before passing them to
        the wrapped module.
        
        Args:
            *args: Positional arguments to pass to the wrapped module
            **kwargs: Keyword arguments to pass to the wrapped module
            
        Returns:
            Output from the wrapped module with appropriate dtype handling
        """
        # Determine the dtype of the module's parameters
        weight_dtype = None
        for name, param in self.module.named_parameters():
            if 'weight' in name:
                weight_dtype = param.dtype
                break
        if weight_dtype is None:
            # If no weight parameter is found, get the dtype of the first parameter
            try:
                weight_dtype = next(self.module.parameters()).dtype
            except StopIteration:
                # If no parameters exist, just return the module's output
                return self.module(*args, **kwargs)

        def to_dtype(x):
            """
            Recursively convert tensors to the target dtype, preserving non-floating types.
            
            Args:
                x: Input to potentially convert
                
            Returns:
                Converted input with appropriate dtypes
            """
            if isinstance(x, torch.Tensor):
                if x.is_floating_point():
                    return x.to(weight_dtype)
                else:
                    return x  # Don't change dtype of non-floating tensors (like indices)
            elif isinstance(x, tuple):
                return tuple(to_dtype(item) for item in x)
            elif isinstance(x, list):
                return [to_dtype(item) for item in x]
            elif isinstance(x, dict):
                return {k: to_dtype(v) for k, v in x.items()}
            else:
                return x

        # Apply dtype conversion to all arguments
        args = [to_dtype(arg) for arg in args]
        kwargs = {k: to_dtype(v) for k, v in kwargs.items()}

        return self.module(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped module if not found in this wrapper.
        
        This allows accessing attributes and methods of the wrapped module directly
        through the wrapper.
        
        Args:
            name (str): Name of the attribute to access
            
        Returns:
            The requested attribute from either this wrapper or the wrapped module
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
