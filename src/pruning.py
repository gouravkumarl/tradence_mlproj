"""
This file implements a custom prunable linear layer and a self-pruning network.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.gate_scores, 0.0)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def gates(self):
        return torch.sigmoid(self.gate_scores)

    def forward(self, x):
        pruned_weights = self.weight * self.gates
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold=1e-2):
        gate_values = self.gates.detach()
        num_pruned = (gate_values < threshold).sum().item()
        total_gates = gate_values.numel()
        return (num_pruned / total_gates) * 100 if total_gates > 0 else 0.0

    def get_gate_loss(self):
        return self.gates.sum()

    def get_gate_values(self):
        return self.gates.detach().clone()


class SelfPruningNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SelfPruningNN, self).__init__()
        self.layer1 = PrunableLinear(input_size, hidden_size)
        self.layer2 = PrunableLinear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

    def get_sparsity_loss(self):
        return self.layer1.get_gate_loss() + self.layer2.get_gate_loss()

    def get_sparsity(self, threshold=1e-2):
        return (self.layer1.get_sparsity(threshold) + self.layer2.get_sparsity(threshold)) / 2

    def get_network_state(self):
        state = {
            'layer1_gates': self.layer1.get_gate_values(),
            'layer2_gates': self.layer2.get_gate_values(),
            'layer1_sparsity': self.layer1.get_sparsity(),
            'layer2_sparsity': self.layer2.get_sparsity(),
            'average_sparsity': self.get_sparsity(),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'layer1_weight_norm': self.layer1.weight.norm().item(),
            'layer2_weight_norm': self.layer2.weight.norm().item(),
        }
        return state
