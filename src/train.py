import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms

from pruning import SelfPruningNN
from utils import (
    calculate_sparsity, 
    evaluate_model, 
    report_network_state,
    count_active_parameters,
    count_parameters
)


def create_dummy_dataset(n_samples=500, input_size=10, output_size=2, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random data
    X = torch.randn(n_samples, input_size)
    # Generate labels based on a simple rule
    y = torch.randint(0, output_size, (n_samples,))
    
    # Split into train and test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


def create_cifar10_loaders(batch_size=128, data_dir='./data'):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, lambda_param, device='cpu'):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Add sparsity regularization loss on gates
        reg_loss = model.get_sparsity_loss()
        total_loss_value = loss + lambda_param * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()
        
        total_loss += total_loss_value.item()
    
    return total_loss / len(train_loader)


def train_model(
    model,
    train_loader,
    test_loader,
    lambda_param,
    num_epochs=50,
    learning_rate=0.001,
    device='cpu',
    verbose=True
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'sparsity': [],
        'active_params': []
    }
    
    # Training loop
    pbar = tqdm(range(num_epochs), disable=not verbose)
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, lambda_param, device)
        history['train_loss'].append(train_loss)
        
        # Evaluate
        test_results = evaluate_model(model, test_loader, criterion, device)
        history['test_loss'].append(test_results['loss'])
        history['test_accuracy'].append(test_results['accuracy'])
        
        # Calculate sparsity
        sparsity = calculate_sparsity(model, threshold=1e-2)
        active_params = count_active_parameters(model, threshold=1e-2)
        total_params = count_parameters(model)
        
        history['sparsity'].append(sparsity)
        history['active_params'].append(active_params)
        
        if verbose:
            pbar.set_description(
                f"λ={lambda_param:.2e} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_results['accuracy']:.2f}% | "
                f"Sparsity: {sparsity:.2f}%"
            )
    
    return history, test_results


def run_lambda_comparison(
    lambda_values,
    input_size=3072,
    hidden_size=64,
    output_size=10,
    num_epochs=50,
    learning_rate=0.001,
    device='cpu',
    use_cifar=True,
    batch_size=128,
    data_dir='./data'
):
    
    # Create datasets (same for all experiments)
    if use_cifar:
        train_loader, test_loader = create_cifar10_loaders(batch_size=batch_size, data_dir=data_dir)
    else:
        train_loader, test_loader = create_dummy_dataset(n_samples=500, input_size=input_size)
    
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"LAMBDA COMPARISON EXPERIMENTS")
    print(f"{'='*70}\n")
    
    for lambda_param in lambda_values:
        print(f"\n{'*'*70}")
        print(f"Training with λ = {lambda_param:.2e}")
        print(f"{'*'*70}\n")
        
        # Create fresh model for each lambda
        model = SelfPruningNN(input_size, hidden_size, output_size)
        model = model.to(device)
        
        # Train
        history, final_results = train_model(
            model,
            train_loader,
            test_loader,
            lambda_param,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            verbose=True
        )
        
        # Store results
        final_sparsity = calculate_sparsity(model, threshold=1e-2)
        final_accuracy = final_results['accuracy']
        
        all_results[lambda_param] = {
            'model': model,
            'history': history,
            'final_accuracy': final_accuracy,
            'final_sparsity': final_sparsity,
            'final_test_loss': final_results['loss']
        }
        
        # Print final state report
        print(report_network_state(model, lambda_param, num_epochs, final_results))
    
    return all_results


def print_final_comparison(all_results):
    """
    Print a comprehensive comparison of all lambda experiments.
    """
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON - SPARSITY vs ACCURACY TRADE-OFF")
    print(f"{'='*70}\n")
    
    print(f"{'Lambda':<20} {'Accuracy (%)':<20} {'Sparsity (%)':<20}")
    print(f"{'-'*60}")
    
    for lambda_val in sorted(all_results.keys()):
        results = all_results[lambda_val]
        acc = results['final_accuracy']
        sparsity = results['final_sparsity']
        print(f"{lambda_val:<20.2e} {acc:<20.2f} {sparsity:<20.2f}")
    
    print(f"{'='*70}\n")
    
    # Find best models
    best_accuracy_lambda = max(all_results.keys(), key=lambda x: all_results[x]['final_accuracy'])
    best_sparsity_lambda = max(all_results.keys(), key=lambda x: all_results[x]['final_sparsity'])
    
    print(f"Best Accuracy: λ = {best_accuracy_lambda:.2e} → {all_results[best_accuracy_lambda]['final_accuracy']:.2f}%")
    print(f"Best Sparsity: λ = {best_sparsity_lambda:.2e} → {all_results[best_sparsity_lambda]['final_sparsity']:.2f}%\n")


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Define lambda values for comparison (low, medium, high)
    lambda_values = [0.0001, 0.001, 0.01]
    
    # Run experiments on CIFAR-10
    results = run_lambda_comparison(
        lambda_values=lambda_values,
        input_size=32 * 32 * 3,
        hidden_size=128,
        output_size=10,
        num_epochs=50,
        learning_rate=0.001,
        device=device,
        use_cifar=True,
        batch_size=128,
        data_dir='./data'
    )
    
    # Print final comparison
    print_final_comparison(results)
