"""
Main Training Script
Supports all W&B logging required for Q2.3 – Q2.10.
"""

import argparse
import json
import os
import numpy as np

from utils.data_loader import load_data
from utils import create_dir
from ann.neural_network import NeuralNetwork
from ann.objective_functions import compute_loss
from ann.optimizers import NAG


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("-d",   "--dataset",      type=str,   default="mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=32)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mse"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+",
                        default=[128, 128, 128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",
                        choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier"])

    # W&B — single declaration, accessible as -w_p or --wandb_project
    parser.add_argument("-w_p", "--wandb_project", type=str,   default=None)
    parser.add_argument("--wandb_entity",          type=str,   default=None)
    parser.add_argument("--run_name",              type=str,   default=None)

    # Extra diagnostic logging flags
    parser.add_argument("--log_grad_norms",   action="store_true")
    parser.add_argument("--log_dead_neurons", action="store_true")
    parser.add_argument("--log_activations",  action="store_true")
    parser.add_argument("--log_neuron_grads", action="store_true")
    parser.add_argument("--zero_init",        action="store_true")

    parser.add_argument("--model_dir", type=str, default="../models")
    return parser.parse_args()


def save_config(args, val_acc, model_path, config_path):
    config = {
        "dataset":       args.dataset,
        "input_dim":     args.input_dim,
        "output_dim":    args.output_dim,
        "num_layers":    args.num_layers,
        "hidden_size":   args.hidden_size,
        "activation":    args.activation,
        "weight_init":   args.weight_init,
        "optimizer":     args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size":    args.batch_size,
        "epochs":        args.epochs,
        "loss":          args.loss,
        "weight_decay":  args.weight_decay,
        "best_val_acc":  round(float(val_acc), 6),
        "model_path":    model_path,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def _log_neuron_grad_iter(model, step, wandb, run):
    log = {"global_step": step}
    for li, layer in enumerate(model.layers[:-1]):
        if layer.grad_W is None:
            continue
        n = min(5, layer.grad_W.shape[1])
        for ni in range(n):
            log[f"neuron_grad_L{li+1}_N{ni+1}"] = float(
                np.linalg.norm(layer.grad_W[:, ni])
            )
    run.log(log)


def _log_epoch(model, epoch, avg_loss, val_loss, train_acc, val_acc,
               args, wandb, run, X_val_sample):
    log = {
        "epoch":      epoch + 1,
        "train_loss": float(avg_loss),
        "val_loss":   float(val_loss),
        "train_acc":  float(train_acc),
        "val_acc":    float(val_acc),
    }

    if args.log_grad_norms:
        for li, layer in enumerate(model.layers[:-1]):
            if layer.grad_W is not None:
                log[f"grad_norm_layer{li+1}"] = float(np.linalg.norm(layer.grad_W))

    if args.log_dead_neurons or args.log_activations:
        _ = model.forward(X_val_sample)
        for li, layer in enumerate(model.layers[:-1]):
            if args.log_dead_neurons:
                log[f"dead_neuron_frac_L{li+1}"] = float(np.mean(layer.A <= 0))
            if args.log_activations:
                log[f"activations_L{li+1}"] = wandb.Histogram(layer.A.flatten())

    if args.log_neuron_grads:
        for li, layer in enumerate(model.layers[:-1]):
            if layer.grad_W is not None:
                n = min(5, layer.grad_W.shape[1])
                for ni in range(n):
                    log[f"neuron_grad_L{li+1}_N{ni+1}"] = float(
                        np.linalg.norm(layer.grad_W[:, ni])
                    )
    run.log(log)


def main():
    args = parse_arguments()
    args.input_dim  = 784
    args.output_dim = 10

    use_wandb = args.wandb_project is not None
    wandb = None
    run   = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        run = wandb.init(
            project = args.wandb_project,
            entity  = args.wandb_entity,
            name    = args.run_name,
            config  = {
                "dataset":       args.dataset,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "loss":          args.loss,
                "optimizer":     args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay":  args.weight_decay,
                "num_layers":    args.num_layers,
                "hidden_size":   args.hidden_size,
                "activation":    args.activation,
                "weight_init":   args.weight_init,
                "zero_init":     args.zero_init,
            },
        )

    print(f"Loading dataset: {args.dataset}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    X_val_sample = X_val[:500]

    print("Building model...")
    model = NeuralNetwork(args)   # pass the whole Namespace — always

    if args.zero_init:
        print("[zero_init] All weights and biases set to 0.0")
        for layer in model.layers:
            layer.W[:] = 0.0
            layer.b[:] = 0.0

    is_nag       = isinstance(model.optimizer, NAG)
    n_samples    = X_train.shape[0]
    best_val_acc = -1.0
    global_step  = 0

    create_dir(args.model_dir)
    model_path  = os.path.join(args.model_dir, "best_model.npy")
    config_path = os.path.join(args.model_dir, "best_config.json")

    print("Starting training...")
    for epoch in range(args.epochs):
        idx  = np.random.permutation(n_samples)
        X_tr = X_train[idx]
        y_tr = y_train[idx]

        epoch_loss  = 0.0
        num_batches = 0

        for start in range(0, n_samples, args.batch_size):
            Xb = X_tr[start : start + args.batch_size]
            yb = y_tr[start : start + args.batch_size]

            if is_nag:
                model.optimizer.apply_lookahead(model.layers)
                yp = model.forward(Xb)
                bl = compute_loss(yb, yp, args.loss)
                model.backward(yb, yp)
                model.optimizer.restore_weights(model.layers)
            else:
                yp = model.forward(Xb)
                bl = compute_loss(yb, yp, args.loss)
                model.backward(yb, yp)

            model.update_weights()
            epoch_loss  += bl
            num_batches += 1
            global_step += 1

            if use_wandb and args.log_neuron_grads and global_step <= 50:
                _log_neuron_grad_iter(model, global_step, wandb, run)

        avg_loss  = epoch_loss / num_batches
        val_acc   = model.evaluate(X_val, y_val)
        train_acc = model.evaluate(X_tr,  y_tr)
        val_loss  = compute_loss(y_val, model.forward(X_val), args.loss)

        print(
            f"Epoch {epoch+1:>3}/{args.epochs} | "
            f"train_loss {avg_loss:.4f} | val_loss {val_loss:.4f} | "
            f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}",
            end="",
        )

        if use_wandb:
            _log_epoch(model, epoch, avg_loss, val_loss, train_acc, val_acc,
                       args, wandb, run, X_val_sample)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(model_path)
            save_config(args, best_val_acc, model_path, config_path)
            print("  ← new best!")
        else:
            print()

    test_acc = model.evaluate(X_test, y_test)
    print(f"\nBest val accuracy : {best_val_acc:.4f}")
    print(f"Test  accuracy    : {test_acc:.4f}")

    if use_wandb:
        run.summary["best_val_acc"] = best_val_acc
        run.summary["test_acc"]     = test_acc
        run.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()