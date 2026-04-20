# Self-Pruning Neural Network

## Overview
This repository implements a self-pruning neural network with a custom PyTorch layer for learned gate-based sparsity. The model is trained and evaluated on CIFAR-10, and the code is structured for repeatable experiments and clear results.

## Setup Instructions
```bash
cd /home/gourav/Desktop/trancedance/self-pruning-neural-network
python3 -m venv venv
source venv/bin/activate
pip install -r self_pruning_nn/requirements.txt
```

## How to Run
```bash
python3 src/train.py
```

This launches an experiment comparing multiple sparsity regularization strengths and prints:
- train/test loss
- test accuracy
- gate sparsity
- final pruning comparison table

## Core Files
- `src/pruning.py` — custom `PrunableLinear` layer and self-pruning model
- `src/train.py` — CIFAR-10 loader, training loop, and lambda comparison experiments
- `src/utils.py` — evaluation utilities, sparsity metrics, and reporting helpers

## Why This Project is Job-Worthy
- Demonstrates custom PyTorch module design
- Shows expertise in sparse model training
- Uses a real computer vision benchmark (CIFAR-10)
- Includes a reproducible experiment pipeline
- Provides clear documentation and a polished delivery
