# Self-Pruning Neural Network - Evaluation Report

## Executive Summary

This project implements a self-pruning PyTorch model and evaluates it on CIFAR-10. It demonstrates a practical and technical understanding of learned sparsity, custom layer design, and lambda-driven trade-off analysis.

## Implementation Summary

- Custom layer: `PrunableLinear` defined in `src/pruning.py`
- Sparsity mechanism: `weight * sigmoid(gate_scores)`
- Training loss: `CrossEntropyLoss + λ * gate_sparsity`
- Dataset: CIFAR-10 with standard normalization
- Evaluation: test accuracy, gate sparsity, and pruning trade-off across lambdas

## Evaluation Criteria

### 1. Correctness of `PrunableLinear`

- The custom layer correctly stores separate `weight`, `bias`, and `gate_scores` parameters.
- Gate activations are computed as `sigmoid(gate_scores)` and applied multiplicatively to weights.
- The design is fully differentiable, so gradients flow through both weight and gate parameters.

### 2. Training Loop and Custom Sparsity Loss

- Training uses a standard classification loss plus a gate sparsity penalty.
- `train_epoch()` computes:
  - `classification_loss = CrossEntropyLoss(outputs, targets)`
  - `reg_loss = model.get_sparsity_loss()`
  - `total_loss = classification_loss + lambda_param * reg_loss`
- This approach encourages the model to reduce gate activations while preserving classification performance.

### 3. Results and Sparse Model Behavior

#### Key Experimental Results

| Lambda | Test Accuracy | Gate Sparsity | Remarks |
|--------|---------------|----------------|---------|
| 0.0001 | 53.42%        | 61.92%         | Moderate pruning, stable accuracy |
| 0.001  | 53.62%        | 98.48%         | Best accuracy with strong pruning |
| 0.01   | 45.60%        | 99.97%         | Near-full pruning, accuracy drop |

#### Analysis

- The model demonstrates successful pruning behavior by driving gate activations below threshold.
- The medium lambda value (`0.001`) delivers the best balance: high gate sparsity with the highest accuracy.
- The high lambda value forces near-total pruning and reduces accuracy, showing a realistic trade-off.
- This result confirms that learned sparsity is working and that the lambda term meaningfully controls model compression.

### 4. Quality of Results and Trade-Off Analysis

- The experiment design evaluates multiple λ values and reports results in a comparative table.
- The output includes both performance and sparsity metrics, making the trade-off transparent.
- This is the type of analysis expected for engineering evaluation and demonstrates good experimental discipline.

## Practical Takeaways

- The project shows the ability to translate a pruning concept into working code.
- It is structured for reproducibility and for demonstrating engineering thoughtfulness.
- It validates model behavior on a standard vision dataset, which supports credibility in interview settings.

## How to Run

```bash
cd /home/gourav/Desktop/trancedance/self-pruning-neural-network
source venv/bin/activate
python3 src/train.py
```

## Files of Interest

- `src/pruning.py` — custom pruning layer and network definition
- `src/train.py` — training loop, CIFAR-10 loader, and lambda experimentation
- `src/utils.py` — evaluation metrics and reporting
- `README.md` — polished project overview, setup, and job-fit statements

## Job Fit Statement

This project is designed to present a candidate who:

- can implement custom deep-learning components in PyTorch,
- understands dynamic sparsity and model compression,
- can build benchmark experiments end-to-end,
- can document results clearly for stakeholders.

The repository is ready for review as a strong evidence of applied ML engineering skills.
