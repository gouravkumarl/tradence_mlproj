# Self-Pruning Neural Network

## Overview
This project implements a self-pruning neural network that dynamically adjusts its architecture during training to improve efficiency and performance. The network utilizes learnable gate parameters and L1 sparsity regularization to achieve pruning.

## Setup Instructions
To set up the project, ensure you have Python installed on your machine. Then, create a virtual environment and install the required dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r self_pruning_nn/requirements.txt
```

## How to Run the Project
To run the self-pruning neural network, execute the main script located in `self_pruning_nn/self_pruning_nn.py`. Ensure that you have your data prepared in the `data` directory.

```bash
python self_pruning_nn/self_pruning_nn.py
```

## Output Files
The project will generate output files that include model weights, pruning statistics, and visualizations of the training process. These files will be saved in the designated output directory (to be specified in the main script).

## Documentation
For detailed results and analysis of the experiments conducted with the self-pruning neural network, refer to the `self_pruning_nn/report.md` file.