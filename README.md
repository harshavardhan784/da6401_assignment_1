# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

**Course:** DA6401 Introduction to Deep Learning  
**Dataset:** MNIST and Fashion-MNIST  
**Framework:** NumPy only (no autograd libraries)

---

## Project Structure

```
da6401_assignment_1/
├── models/                    # Saved weights and configs (auto-created)
│   ├── best_model.npy
│   └── best_config.json
├── notebooks/                 # Optional Jupyter notebooks
├── src/
│   ├── ann/                   # Core neural network module
│   │   ├── __init__.py
│   │   ├── activations.py     # ReLU, Sigmoid, Tanh, Softmax + derivatives
│   │   ├── neural_layer.py    # Single fully-connected layer (forward + backward)
│   │   ├── neural_network.py  # Full MLP: build, train, evaluate, save/load
│   │   ├── objective_functions.py  # Cross-entropy and MSE losses + gradients
│   │   └── optimizers.py      # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py        # create_dir, set_seed, timestamp helpers
│   │   └── data_loader.py     # MNIST / Fashion-MNIST loading, split, normalise
│   ├── data_exploration.py    # Q2.1 — W&B data exploration script
│   ├── error_analysis.py      # Q2.8 — confusion matrix + failure visualisation
│   ├── inference.py           # Load model, evaluate, print metrics
│   ├── sweep.py               # Q2.2 — Bayesian hyperparameter sweep (100 runs)
│   └── train.py               # Main training script with all W&B logging flags
├── .gitignore
├── commands.md                # All CLI commands for every experiment
├── README.md
└── requirements.txt
```

---

## Setup

```bash
git clone ..
cd da6401_assignment_1

pip install -r requirements.txt
wandb login          # paste key from wandb.ai/authorize

cd src
```

---

## Requirements

```
numpy>=1.21.0
matplotlib>=3.4.0
keras>=2.7.0
wandb>=0.12.0
scikit-learn>=0.24.2
```

---

## Train

```bash
# Basic run (no W&B logging)
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy

# With W&B logging
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name my_run
```

### CLI Arguments

| Flag | Long form | Description | Default |
|------|-----------|-------------|---------|
| `-d` | `--dataset` | `mnist` or `fashion_mnist` | `mnist` |
| `-e` | `--epochs` | Number of training epochs | `10` |
| `-b` | `--batch_size` | Mini-batch size | `32` |
| `-l` | `--loss` | `cross_entropy` or `mse` | `cross_entropy` |
| `-o` | `--optimizer` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` | `adam` |
| `-lr` | `--learning_rate` | Initial learning rate | `1e-3` |
| `-wd` | `--weight_decay` | L2 regularisation coefficient | `0.0` |
| `-nhl` | `--num_layers` | Number of hidden layers | `3` |
| `-sz` | `--hidden_size` | Neurons per hidden layer (space-separated) | `128 128 128` |
| `-a` | `--activation` | `relu`, `sigmoid`, or `tanh` | `relu` |
| `-w_i` | `--weight_init` | `random` or `xavier` | `xavier` |

### W&B Extra Logging Flags

| Flag | Used for |
|------|----------|
| `--log_grad_norms` | Q2.4 — gradient norms per hidden layer per epoch |
| `--log_dead_neurons` | Q2.5 — fraction of zero-output neurons per layer |
| `--log_activations` | Q2.5 — activation histograms per layer |
| `--log_neuron_grads` | Q2.9 — per-neuron grad norms (first 50 iterations + per epoch) |
| `--zero_init` | Q2.9 — set all weights and biases to zero after construction |

---

## Inference

```bash
python -m inference --model_path ../models/best_model.npy \
    -d mnist -nhl 3 -sz 128 128 128 -a relu
```

Outputs: **Accuracy, Precision, Recall, F1-score** (macro-averaged).

---

## Hyperparameter Sweep (Q2.2)

```bash
python -m sweep --wandb_project da6401-a1 --count 100
```

Runs a 100-trial Bayesian sweep over: learning rate, batch size, number of layers, hidden size, activation, optimizer, weight decay.

---

## Data Exploration (Q2.1)

```bash
python -m data_exploration --wandb_project da6401-a1
```

Logs to W&B: sample image table (5 per class), class-distribution bar chart, per-class mean-image grid, and inter-class cosine-similarity heatmap.

---

## Error Analysis (Q2.8)

```bash
python -m error_analysis \
    --model_path ../models/best_model.npy \
    --wandb_project da6401-a1 \
    -d mnist -nhl 3 -sz 128 128 128 -a relu
```

Logs: standard confusion matrix, per-class accuracy bar chart, and a grid of the most confidently mis-classified examples per class.

---

## Implementation Notes

### Gradient correctness
All gradients are computed analytically in `NeuralLayer.backward()` and exposed as `self.grad_W` and `self.grad_b` after every call. The cross-entropy + softmax combined gradient `(ŷ − y)` is used directly for the output layer when `loss=cross_entropy`, avoiding a redundant Jacobian multiplication. For `loss=mse`, the full softmax Jacobian-vector product is applied correctly.

### Data pipeline
- Train / validation split: 90 / 10, stratified, using `sklearn.model_selection.train_test_split`.
- Normalisation uses **training-set statistics only** (mean and std per pixel), applied identically to validation and test sets. No data leakage.

### Weight initialisation
- `random`: Gaussian with σ = 0.01.
- `xavier`: Gaussian scaled by `sqrt(2 / (fan_in + fan_out))`.

### Optimizers
All six optimizers are implemented from scratch with NumPy:
- **SGD** — vanilla gradient descent.
- **Momentum** — exponential moving average of gradients.
- **NAG** — true Nesterov look-ahead: weights are temporarily shifted before the gradient is computed, then restored before the actual update.
- **RMSProp** — per-parameter adaptive scaling using a running average of squared gradients.
- **Adam** — bias-corrected first and second moment estimates. Timestep `t` is tracked **per layer** (not globally) to give correct bias correction regardless of network depth.
- **Nadam** — Nesterov-corrected Adam with look-ahead applied to the moment estimate.

---

## W&B Report

Public report link: `https://wandb.ai/harshavardhan-govind-iit-madras/da6401-a1/reports/DA6401_A1--VmlldzoxNjA0MDQyMQ?accessToken=n6c21h9a4s09irursmtbakstof8luxpidunymuazi2q743hxccreej1xc7ti8q53`

The report covers questions Q2.1 through Q2.10 as specified in the assignment.

---

## Results

| Dataset | Best Train accuracy | Val accuracy |
|---------|-------------------|---------------|
| MNIST | 0.99 | 0.97 |
| Fashion-MNIST | 0.91 | 0.89 |

---