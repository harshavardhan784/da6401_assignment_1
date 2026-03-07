# DA6401 Assignment 1 — Multi-Layer Perceptron for Image Classification

**Course:** DA6401 Introduction to Deep Learning  
**Dataset:** MNIST and Fashion-MNIST  
**Framework:** NumPy only (no autograd libraries)

---

## 🔗 Links

| Resource | Link |
|----------|------|
| **W&B Report** | [View Report](https://wandb.ai/harshavardhan-govind-iit-madras/da6401-a1/reports/da6401-a1--VmlldzoxNjEyNjIzMw?accessToken=2w45fxedum2xzy2mwuc1erifnaau3jmw7e3lnryw58rfs2hn2ohpy9ny2wb0xu5g) |
| **GitHub Repo** | [da6401_assignment_1](https://github.com/harshavardhan784/da6401_assignment_1.git) |

---

## Project Structure

```
da6401_assignment_1/
├── models/
│   ├── best_model.npy
│   └── best_config.json
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── neural_layer.py
│   │   ├── neural_network.py
│   │   ├── objective_functions.py
│   │   └── optimizers.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── data_exploration.py
│   ├── error_analysis.py
│   ├── inference.py
│   ├── sweep.py
│   └── train.py
├── .gitignore
├── commands.md
├── README.md
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/harshavardhan784/da6401_assignment_1.git
cd da6401_assignment_1

pip install -r requirements.txt
wandb login

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
python -m train -d mnist -e 10 -b 32 -o rmsprop -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy

# With W&B logging
python -m train -d mnist -e 10 -b 32 -o rmsprop -lr 1e-3 \
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
| `-o` | `--optimizer` | `sgd`, `momentum`, `nag`, `rmsprop` | `rmsprop` |
| `-lr` | `--learning_rate` | Initial learning rate | `1e-3` |
| `-wd` | `--weight_decay` | L2 regularisation coefficient | `0.0` |
| `-nhl` | `--num_layers` | Number of hidden layers | `3` |
| `-sz` | `--hidden_size` | Neurons per hidden layer (space-separated) | `128 128 128` |
| `-a` | `--activation` | `relu`, `sigmoid`, or `tanh` | `relu` |
| `-w_i` | `--weight_init` | `random` or `xavier` | `xavier` |
| `-w_p` | `--wandb_project` | W&B project name | `None` |

### W&B Extra Logging Flags

| Flag | Used for |
|------|----------|
| `--log_grad_norms` | Q2.4 — gradient norms per hidden layer per epoch |
| `--log_dead_neurons` | Q2.5 — fraction of zero-output neurons per layer |
| `--log_activations` | Q2.5 — activation histograms per layer |
| `--log_neuron_grads` | Q2.9 — per-neuron grad norms (first 50 iterations) |
| `--zero_init` | Q2.9 — set all weights and biases to zero |

---

## Inference

```bash
python -m inference --model_path best_model.npy \
    -d mnist -nhl 3 -sz 128 128 128 -a relu -o rmsprop
```

Outputs: **Accuracy, Precision, Recall, F1-score** (macro-averaged).

---

## Implementation Notes

### Optimizers
All four optimizers are implemented from scratch with NumPy:
- **SGD** — vanilla gradient descent (processes mini-batches).
- **Momentum** — exponential moving average of gradients.
- **NAG** — true Nesterov look-ahead: weights are temporarily shifted before the gradient is computed, then restored before the actual update.
- **RMSProp** — per-parameter adaptive scaling using a running average of squared gradients.

### Gradient correctness
All gradients are computed analytically in `NeuralLayer.backward()` and exposed as `self.grad_W` and `self.grad_b` after every call. The cross-entropy + softmax combined gradient `(ŷ − y)` is used directly for the output layer, avoiding a redundant Jacobian multiplication.

### Data pipeline
- Train / validation split: 90 / 10, stratified.
- Normalisation: pixel values divided by 255 to scale to [0, 1], using training-set statistics only. No data leakage.

### Weight initialisation
- `random`: Gaussian with σ = 0.01.
- `xavier`: Gaussian scaled by `sqrt(2 / (fan_in + fan_out))`.

---

## Results

| Dataset | Best Train Accuracy | Val Accuracy |
|---------|-------------------|---------------|
| MNIST | 0.99 | 0.97 |
| Fashion-MNIST | 0.91 | 0.89 |