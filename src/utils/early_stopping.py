import torch
import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0):
        """
        Initialize the EarlyStopping object.
        
        Args:
            accelerator: Accelerator object for multi-GPU training (e.g., Hugging Face Accelerate)
            patience (int): Number of epochs to wait before early stopping once no improvement is observed
            verbose (bool): Whether to print messages about early stopping
            delta (float): Minimum change in validation loss to be considered as improvement
        """
        self.accelerator = accelerator  # Accelerator for distributed training
        self.patience = patience        # Number of epochs to wait before stopping
        self.verbose = verbose          # Whether to print status updates
        self.counter = 0                # Counter for epochs without improvement
        self.best_score = None          # Best validation score seen so far
        self.early_stop = False         # Flag to indicate if training should stop
        self.val_loss_min = np.Inf      # Minimum validation loss seen so far
        self.delta = delta              # Minimum change to qualify as improvement

    def __call__(self, val_loss, model, path):
        """
        Call method to check if validation loss has improved and update early stopping status.
        
        Args:
            val_loss: Current validation loss
            model: Model to save if validation loss improves
            path: Path to save the best model checkpoint
        """
        # Convert loss to score (higher is better, so we use negative loss)
        score = -val_loss
        
        # If this is the first call, set the best score and save the model
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        # If the current score is not better than the best score minus delta
        elif score < self.best_score - self.delta * self.best_score:
            self.counter += 1  # Increment the counter for epochs without improvement

            # Print status update
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            # If patience is exceeded, set early stop flag
            if self.counter >= self.patience:
                self.early_stop = True
        # If the current score is better, update best score and reset counter
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0  # Reset the counter

    def save_checkpoint(self, val_loss, model, path):
        """
        Save the model checkpoint when validation loss decreases.
        
        Args:
            val_loss: Current validation loss
            model: Model to save
            path: Path to save the checkpoint
        """
        # Print message if verbose is enabled
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Save the model state dict to the specified path
        # Note: The commented code shows an alternative implementation
        # if self.accelerator is not None:
        #     model = self.accelerator.unwrap_model(model)
        #     torch.save(model.state_dict(), path + '/' + 'checkpoint')
        # else:
        #     torch.save(model.state_dict(), path + '/' + 'checkpoint')
        
        # Update the minimum validation loss
        self.val_loss_min = val_loss

        # Save only trainable parameters to reduce checkpoint size
        trainable_state_dict = {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}
        
        # Use accelerator to save the model if available
        if self.accelerator is not None:
            self.accelerator.save(trainable_state_dict, path + '/' + 'checkpoint.pt')
        else:
            # Fallback to torch.save if no accelerator is provided
            torch.save(trainable_state_dict, path + '/' + 'checkpoint.pt')