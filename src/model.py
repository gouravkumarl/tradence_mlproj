import torch
import torch.nn as nn
from .pruning import SelfPruningNN

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def create_model(input_size, hidden_size, output_size, use_pruning=True):
    """
    Create a neural network model.
    
    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        output_size: Size of output
        use_pruning: If True, use SelfPruningNN; otherwise use SimpleNN
    
    Returns:
        model: Neural network model
    """
    if use_pruning:
        model = SelfPruningNN(input_size, hidden_size, output_size)
    else:
        model = SimpleNN(input_size, hidden_size, output_size)
    return model
