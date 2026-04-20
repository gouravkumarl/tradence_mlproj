import torch
import torch.nn as nn
from .pruning import SelfPruningNN


def create_model(input_size, hidden_size, output_size):
    """
    Create a self-pruning neural network model using custom PrunableLinear layers.
    
    This function creates the required custom prunable model architecture.
    Standard torch.nn.Linear layers are NOT used - instead, PrunableLinear layers
    with learnable gate_scores are employed for sparsity regularization.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        output_size: Size of output
    
    Returns:
        model: SelfPruningNN model with custom PrunableLinear layers
    """
    model = SelfPruningNN(input_size, hidden_size, output_size)
    return model
