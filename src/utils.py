import torch
from sklearn.metrics import accuracy_score

def evaluate_model(model, dataloader, criterion, device='cpu'):
    """Evaluate model on test set."""
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
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds) * 100
    return {'loss': total_loss / len(dataloader), 'accuracy': accuracy}

def calculate_sparsity(model, threshold=1e-2):
    """Calculate gate sparsity percentage."""
    total_gates = 0
    total_pruned = 0
    
    for layer in [model.layer1, model.layer2]:
        gates = layer.gates.detach()
        total_gates += gates.numel()
        total_pruned += (gates < threshold).sum().item()
    
    return (total_pruned / total_gates) * 100 if total_gates > 0 else 0.0