# DA6401 Assignment 1 — Run Commands
# All commands run from: da6401_assignment_1/src/

---

## Setup
```bash
pip install -r requirements.txt
wandb login
cd src
```

---

## Part 1 — Train & Inference

```bash
# Basic train (no W&B)
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy

# With W&B
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name best_run

# Inference on saved model
python -m inference --model_path ../models/best_model.npy \
    -d mnist -nhl 3 -sz 128 128 128 -a relu
```

---

## Part 2 — W&B Report

### Q2.1 — Data Exploration & Class Distribution
Logs: sample images table, class-distribution bar chart,
      per-class mean-image grid, inter-class similarity heatmap.

```bash
# Both MNIST and Fashion-MNIST
python -m data_exploration --wandb_project da6401-a1

# Single dataset
python -m data_exploration --wandb_project da6401-a1 --dataset fashion_mnist
```

---

### Q2.2 — Hyperparameter Sweep (100 runs, Bayesian)
```bash
python -m sweep --wandb_project da6401-a1 --count 100
```

---

### Q2.3 — Optimizer Showdown (same architecture, all 6 optimizers)
Fixed: 3 hidden layers × 128 neurons, ReLU, Xavier, CE loss, lr=1e-3.

Windows:
```powershell
foreach ($OPT in @("sgd","momentum","nag","rmsprop","adam","nadam")) {
  python -m train -d mnist -e 10 -b 32 -o $OPT -lr 1e-3 `
      -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy `
      --wandb_project da6401-a1 --run_name "optimizer_$OPT"
}
```


---

### Q2.4 — Vanishing Gradient Analysis (sigmoid vs ReLU, 3L and 5L)
Uses --log_grad_norms → logs ||grad_W|| for each hidden layer per epoch.

```bash
# 3 hidden layers
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name vanishing_relu_3L \
    --log_grad_norms

python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a sigmoid -w_i xavier \
    --wandb_project da6401-a1 --run_name vanishing_sigmoid_3L \
    --log_grad_norms

# 5 hidden layers (deeper → more pronounced vanishing)
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 5 -sz 128 128 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name vanishing_relu_5L \
    --log_grad_norms

python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 5 -sz 128 128 128 128 128 -a sigmoid -w_i xavier \
    --wandb_project da6401-a1 --run_name vanishing_sigmoid_5L \
    --log_grad_norms
```

---

### Q2.5 — Dead Neuron Investigation (ReLU high-lr vs normal vs Tanh)
Uses --log_dead_neurons (fraction of zero-output neurons per layer)
and  --log_activations  (histogram of layer activations).

```bash
# High LR → dead ReLU neurons
python -m train -d mnist -e 10 -b 32 -o adam -lr 0.1 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name dead_relu_lr0.1 \
    --log_dead_neurons --log_activations

# Normal LR → healthy ReLU
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name dead_relu_lr1e-3 \
    --log_dead_neurons --log_activations

# Tanh (never truly dead — compare gradients)
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a tanh -w_i xavier \
    --wandb_project da6401-a1 --run_name dead_tanh_lr1e-3 \
    --log_dead_neurons --log_activations
```

---

### Q2.6 — Loss Function Comparison (Cross-Entropy vs MSE)
Same architecture and learning rate; only --loss differs.

```bash
python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name loss_ce

python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l mse \
    --wandb_project da6401-a1 --run_name loss_mse
```

---

### Q2.7 — Global Performance Analysis (train acc vs test acc, all runs)
Run a few extra configs to populate the scatter plot; all previous runs
already count.

```bash
python -m train -d mnist -e 10 -b 64 -o sgd -lr 1e-2 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name global_sgd_lr1e-2

python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name global_adam_4L

python -m train -d mnist -e 10 -b 32 -o nadam -lr 1e-3 -wd 1e-4 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name global_nadam_wd

python -m train -d mnist -e 10 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 64 64 64 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name global_adam_64x3
```

---

### Q2.8 — Error Analysis (Confusion Matrix + Creative Failure Visualization)
Logs: wandb confusion matrix, per-class accuracy bar chart,
      and a misclassified-examples grid (true class × top-5 confident failures).

```bash
# Standard error analysis on best saved model
python -m error_analysis \
    --model_path ../models/best_model.npy \
    --wandb_project da6401-a1 \
    -d mnist -nhl 3 -sz 128 128 128 -a relu

# Fashion-MNIST version (adjust architecture to match your best Fashion model)
python -m error_analysis \
    --model_path ../models/best_model.npy \
    --wandb_project da6401-a1 \
    -d fashion_mnist -nhl 4 -sz 128 128 128 128 -a relu
```
> W&B UI → run "error_analysis" → Charts tab:
>   • confusion_matrix — standard heatmap
>   • per_class_accuracy — bar chart (red = below 90%)
>   • misclassified_examples — grid of worst failures with predicted class + confidence

---

### Q2.9 — Weight Initialization & Symmetry Breaking
Uses --log_neuron_grads → logs per-neuron grad norms for first 5 neurons in
every hidden layer at EVERY ITERATION for the first 50 iterations (then per-epoch).

Xavier run — gradients diverge immediately (symmetry broken):
```bash
python -m train -d mnist -e 5 -b 32 -o sgd -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name init_xavier \
    --log_neuron_grads
```

Zeros run — ALL 5 neuron lines overlap perfectly (symmetry unbroken):
```bash
python -m train -d mnist -e 5 -b 32 -o sgd -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier \
    --wandb_project da6401-a1 --run_name init_zeros \
    --log_neuron_grads --zero_init
```


---

### Q2.10 — Fashion-MNIST Transfer (3 best configs from MNIST learnings)
Based on MNIST sweeps, pick 3 configurations:
  1. Adam + ReLU + 3L×128   (best overall MNIST performer)
  2. Nadam + ReLU + 4L×128 + weight-decay  (regularized deeper net)
  3. Adam + Tanh + 3L×128   (compare smooth activation on harder dataset)

```bash
python -m train -d fashion_mnist -e 15 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a relu -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name fashion_adam_relu_3L

python -m train -d fashion_mnist -e 15 -b 32 -o nadam -lr 1e-3 -wd 1e-4 \
    -nhl 4 -sz 128 128 128 128 -a relu -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name fashion_nadam_relu_4L_wd

python -m train -d fashion_mnist -e 15 -b 32 -o adam -lr 1e-3 \
    -nhl 3 -sz 128 128 128 -a tanh -w_i xavier -l cross_entropy \
    --wandb_project da6401-a1 --run_name fashion_adam_tanh_3L
```