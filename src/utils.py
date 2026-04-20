"""
Utility functions for data loading, metrics calculation, and results reporting.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os

def load_data(data_path, test_size=0.2, random_state=42):
    """
    Load data from CSV file and split into train/test sets.
    
    Args:
        data_path: Path to CSV file
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(data_path)
    
    # Separate features and target (assume last column is target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler


def create_synthetic_data(num_samples=500, num_features=10, num_classes=2, random_state=42):
    """
    Create synthetic regression/classification data for testing.
    
    Args:
        num_samples: Number of samples
        num_features: Number of features
        num_classes: Number of output classes
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    X = np.random.randn(num_samples, num_features)
    if num_classes > 1:
        y = np.random.randint(0, num_classes, num_samples)
    else:
        y = np.random.randn(num_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def calculate_metrics(y_true, y_pred, is_classification=True):
    """
    Calculate various metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels/values
        is_classification: If True, calculate classification metrics; else regression
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    if is_classification:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
    
    return metrics


def save_results(results, output_dir='./results'):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary containing experiment results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'experiment_results.json')
    
    # Convert tensors to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_serializable[key] = {
                k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v
                for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            results_serializable[key] = [
                float(v) if isinstance(v, (torch.Tensor, np.floating)) else v
                for v in value
            ]
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to {output_path}")


def print_results_summary(results, lambda_values):
    """
    Print a formatted summary of experiment results.
    
    Args:
        results: Dictionary containing results for each lambda value
        lambda_values: List of lambda values used in experiments
    """
    print("\n" + "="*80)
    print("SELF-PRUNING NEURAL NETWORK - EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Lambda Value':<15} {'Sparsity (%)':<20} {'Test Accuracy':<20} {'Test Loss':<15}")
    print("-"*70)
    
    for i, lambda_val in enumerate(lambda_values):
        sparsity = results[f'lambda_{lambda_val}']['sparsity']
        accuracy = results[f'lambda_{lambda_val}']['test_accuracy']
        loss = results[f'lambda_{lambda_val}']['final_test_loss']
        print(f"{lambda_val:<15.4f} {sparsity:<20.2f} {accuracy:<20.4f} {loss:<15.6f}")
    
    print("-"*70)
    print("\nNetwork Architecture: 2-layer Self-Pruning Neural Network")
    print("Evaluation Metrics: Sparsity Level | Test Accuracy | Final Test Loss")
    print("="*80 + "\n")


def print_network_state(network_state, lambda_val):
    """
    Print detailed network state information.
    
    Args:
        network_state: Dictionary containing network state
        lambda_val: Lambda value used in training
    """
    print(f"\nFinal Network State (λ = {lambda_val}):")
    print("-" * 60)
    print(f"Average Sparsity Level: {network_state['average_sparsity']:.2f}%")
    print(f"Layer 1 Sparsity: {network_state['layer1_sparsity']:.2f}%")
    print(f"Layer 2 Sparsity: {network_state['layer2_sparsity']:.2f}%")
    print(f"Total Network Parameters: {network_state['total_parameters']}")
    print(f"Layer 1 Weight Norm: {network_state['layer1_weight_norm']:.6f}")
    print(f"Layer 2 Weight Norm: {network_state['layer2_weight_norm']:.6f}")
    print("-" * 60)

def calculate_sparsity(model, threshold=1e-2):
    """
    Calculate the sparsity level of the network.
    Sparsity is the percentage of gate values below the threshold.
    
    Args:
        model: SelfPruningNN model
        threshold: Gate value threshold for sparsity calculation
    
    Returns:
        float: Sparsity percentage
    """
    total_gates = 0
    total_pruned = 0
    
    for module in model.modules():
        gate_values = None
        if hasattr(module, 'gates'):
            gate_values = module.gates.detach()
        elif hasattr(module, 'gate'):
            gate_values = torch.abs(module.gate).detach()
        elif hasattr(module, 'gate_scores'):
            gate_values = torch.sigmoid(module.gate_scores).detach()

        if gate_values is not None:
            total_gates += gate_values.numel()
            total_pruned += (gate_values < threshold).sum().item()
    
    if total_gates == 0:
        return 0.0
    
    sparsity = (total_pruned / total_gates) * 100
    return sparsity


def calculate_weight_sparsity(model, threshold=1e-2):
    """
    Calculate the sparsity of effective weights (after gate scaling).
    
    Args:
        model: SelfPruningNN model
        threshold: Weight value threshold
    
    Returns:
        float: Weight sparsity percentage
    """
    total_weights = 0
    total_zero_weights = 0
    
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if hasattr(module, 'gate_scores'):
                gates = torch.sigmoid(module.gate_scores).detach()
                weights = torch.abs(module.weight * gates).detach()
            else:
                weights = torch.abs(module.weight).detach()
            total_weights += weights.numel()
            total_zero_weights += (weights < threshold).sum().item()
    
    if total_weights == 0:
        return 0.0
    
    sparsity = (total_zero_weights / total_weights) * 100
    return sparsity


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_parameters(model, threshold=1e-2):
    """
    Count parameters that are 'active' (weights or gates above threshold).
    """
    active_params = 0
    for module in model.modules():
        if hasattr(module, 'gates'):
            gates = module.gates.detach()
            active_params += (gates >= threshold).sum().item()
        elif isinstance(module, torch.nn.Linear):
            weights = torch.abs(module.weight).detach()
            active_params += (weights >= threshold).sum().item()
        elif hasattr(module, 'gate'):
            gates = torch.abs(module.gate).detach()
            active_params += (gates >= threshold).sum().item()
    
    return active_params


def evaluate_model(model, dataloader, criterion, device='cpu'):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural network model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: CPU or CUDA device
    
    Returns:
        dict: Contains loss, accuracy, and other metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # For classification
            if outputs.shape[1] > 1:  # Multi-class
                preds = torch.argmax(outputs, dim=1)
            else:  # Binary
                preds = (outputs > 0.5).long().squeeze()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy * 100,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results


def report_network_state(model, lambda_param, epoch, results):
    """
    Generate a comprehensive report of the network state.
    
    Args:
        model: SelfPruningNN model
        lambda_param: L1 regularization parameter
        epoch: Current epoch
        results: Dictionary containing evaluation results
    
    Returns:
        str: Formatted report
    """
    sparsity = calculate_sparsity(model, threshold=1e-2)
    weight_sparsity = calculate_weight_sparsity(model, threshold=1e-2)
    total_params = count_parameters(model)
    active_params = count_active_parameters(model, threshold=1e-2)
    pruning_ratio = (1 - active_params / total_params) * 100 if total_params > 0 else 0
    
    report = f"""
{'='*70}
NETWORK STATE REPORT
{'='*70}
Epoch: {epoch}
Lambda (L1 Regularization): {lambda_param:.2e}

SPARSITY METRICS:
  - Gate Sparsity: {sparsity:.2f}%
  - Weight Sparsity: {weight_sparsity:.2f}%
  - Pruning Ratio: {pruning_ratio:.2f}%

PARAMETER STATISTICS:
  - Total Parameters: {total_params:,}
  - Active Parameters: {active_params:,}
  - Pruned Parameters: {total_params - active_params:,}

PERFORMANCE METRICS:
  - Loss: {results.get('loss', 0):.6f}
  - Accuracy: {results.get('accuracy', 0):.2f}%

{'='*70}
    """
    
    return report