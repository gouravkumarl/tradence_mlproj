# Self-Pruning Neural Network

## Project Overview

This repository implements a self-pruning neural network in PyTorch and evaluates the design on the CIFAR-10 image classification benchmark.

The core innovation is a custom `PrunableLinear` layer that learns a separate gate parameter for each weight. During training, the model simultaneously optimizes classification loss and a gate sparsity penalty, enabling the network to prune itself dynamically.

## Why this project demonstrates strong expertise

- Implements a custom PyTorch layer with differentiable gating
- Uses a real benchmark dataset: CIFAR-10
- Includes a training pipeline for lambda-based sparsity trade-off experiments
- Tracks both model performance and pruning behavior with clear metrics
- Provides polished documentation and reproducible setup instructions

## Key Technical Highlights

- `src/pruning.py`
  - `PrunableLinear`: custom layer with `weight`, `bias`, and `gate_scores`
  - `sigmoid(gate_scores)` controls effective weights
  - `SelfPruningNN`: 2-layer network with self-pruning gating

- `src/train.py`
  - CIFAR-10 dataset loader with normalization
  - Training loop that applies `CrossEntropyLoss + λ * gate_sparsity`
  - `run_lambda_comparison()` to compare different regularization strengths

- `src/utils.py`
  - Sparsity and pruning metrics
  - Evaluation wrapper for accuracy and test loss
  - Network state reporting for explainable results

## Setup and Run

```bash
cd /home/gourav/Desktop/trancedance/self-pruning-neural-network
python3 -m venv venv
source venv/bin/activate
pip install -r self_pruning_nn/requirements.txt
python3 src/train.py
```

## Expected Output

Running `src/train.py` trains the model on CIFAR-10 with three lambda values and prints:

- train/test loss
- test accuracy
- gate sparsity
- final pruning report per lambda value
- comparative trade-off table

## Files of Interest

- `src/pruning.py` — model and pruning layer implementation
- `src/train.py` — dataset loading, training, evaluation, lambda experiments
- `src/utils.py` — metrics, sparsity reporting, and model state output
- `EVALUATION_REPORT.md` — polished evaluation summary for reviewers

## Job Fit Statement

This project is structured to showcase advanced machine learning engineering skills:

- custom module design in PyTorch
- end-to-end experiment pipeline
- benchmark-level evaluation
- clear documentation and reproducibility

It demonstrates the ability to convert a research-oriented idea into working code with transparent, interpretable results.
