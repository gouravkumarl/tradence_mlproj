import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pruning import SelfPruningNN
from .utils import evaluate_model, calculate_sparsity

def create_cifar10_loaders(batch_size=128, data_dir='./data'):
    """Create CIFAR-10 train and test loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_epoch(model, train_loader, optimizer, criterion, lambda_param, device='cpu'):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        reg_loss = model.get_sparsity_loss()
        total_loss_value = loss + lambda_param * reg_loss
        
        optimizer.zero_grad()
        total_loss_value.backward()
        optimizer.step()
        
        total_loss += total_loss_value.item()
    
    return total_loss / len(train_loader)

def train_model(model, train_loader, test_loader, lambda_param, num_epochs=50, device='cpu'):
    """Train model with given lambda."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in tqdm(range(num_epochs), desc=f"λ={lambda_param:.2e}"):
        train_epoch(model, train_loader, optimizer, criterion, lambda_param, device)
        results = evaluate_model(model, test_loader, criterion, device)
    
    return results

def run_experiments(lambda_values, num_epochs=50, device='cpu'):
    """Run pruning experiments with different lambda values."""
    train_loader, test_loader = create_cifar10_loaders(batch_size=128, data_dir='./data')
    all_results = {}
    
    print(f"\n{'='*70}")
    print(f"SELF-PRUNING NEURAL NETWORK - LAMBDA COMPARISON")
    print(f"{'='*70}\n")
    
    for lambda_param in lambda_values:
        print(f"\nTraining with λ = {lambda_param:.2e}")
        model = SelfPruningNN(input_size=32*32*3, hidden_size=128, output_size=10)
        model = model.to(device)
        
        results = train_model(model, train_loader, test_loader, lambda_param, num_epochs, device)
        sparsity = calculate_sparsity(model, threshold=1e-2)
        
        all_results[lambda_param] = {
            'accuracy': results['accuracy'],
            'sparsity': sparsity
        }
        
        print(f"  Final Accuracy: {results['accuracy']:.2f}%")
        print(f"  Gate Sparsity: {sparsity:.2f}%")
    
    return all_results

def print_comparison(all_results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print(f"{'Lambda':<20} {'Accuracy (%)':<20} {'Sparsity (%)':<20}")
    print(f"{'='*70}")
    
    for lambda_val in sorted(all_results.keys()):
        acc = all_results[lambda_val]['accuracy']
        sp = all_results[lambda_val]['sparsity']
        print(f"{lambda_val:<20.2e} {acc:<20.2f} {sp:<20.2f}")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    lambda_values = [0.0001, 0.001, 0.01]
    results = run_experiments(lambda_values, num_epochs=50, device=device)
    print_comparison(results)
