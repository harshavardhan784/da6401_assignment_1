"""
Main Training Script
Entry point for training neural networks with command-line arguments.
Automatically saves best_model.npy and best_config.json based on
best validation accuracy observed during training.
"""

import argparse
import json
import os
import numpy as np

from utils.data_loader import load_data
from utils import create_dir
from ann.neural_network import NeuralNetwork


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help="Dataset to train on")

    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')

    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Mini-batch size')

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function')

    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate')

    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='Weight decay for L2 regularization')

    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers')

    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128],
                        help='Number of neurons in each hidden layer')

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for hidden layers')

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method')

    parser.add_argument('--wandb_project', type=str, default='da6401-a1',
                        help='W&B project name')

    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Directory to save best_model.npy and best_config.json')

    return parser.parse_args()


def save_config(args, val_acc, model_path, config_path):
    """Serialise the CLI args + best val accuracy to a JSON file."""
    config = {
        "dataset":        args.dataset,
        "input_dim":      args.input_dim,
        "output_dim":     args.output_dim,
        "num_layers":     args.num_layers,
        "hidden_size":    args.hidden_size,
        "activation":     args.activation,
        "weight_init":    args.weight_init,
        "optimizer":      args.optimizer,
        "learning_rate":  args.learning_rate,
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "loss":           args.loss,
        "weight_decay":   args.weight_decay,
        "best_val_acc":   round(float(val_acc), 6),
        "model_path":     model_path,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def main():
    args = parse_arguments()
    print(args)

    # Fixed dimensions for MNIST / Fashion-MNIST
    args.input_dim  = 784
    args.output_dim = 10

    # Paths
    create_dir(args.model_dir)
    model_path  = os.path.join(args.model_dir, "best_model.npy")
    config_path = os.path.join(args.model_dir, "best_config.json")

    print(f"Loading dataset: {args.dataset}")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)

    print("Building model...")
    model = NeuralNetwork(args)

    n_samples   = X_train.shape[0]
    best_val_acc = -1.0

    print("Starting training...")
    for epoch in range(args.epochs):
        # --- shuffle ---
        idx = np.random.permutation(n_samples)
        X_train = X_train[idx]
        y_train = y_train[idx]

        epoch_loss  = 0.0
        num_batches = 0

        for start in range(0, n_samples, args.batch_size):
            X_batch = X_train[start : start + args.batch_size]
            y_batch = y_train[start : start + args.batch_size]

            from ann.objective_functions import compute_loss
            y_pred     = model.forward(X_batch)
            batch_loss = compute_loss(y_batch, y_pred, args.loss)
            model.backward(y_batch, y_pred)
            model.update_weights()

            epoch_loss  += batch_loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        val_acc  = model.evaluate(X_val, y_val)

        print(f"Epoch {epoch + 1}/{args.epochs}  |  loss: {avg_loss:.4f}  |  val_acc: {val_acc:.4f}", end="")

        # --- save best checkpoint ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(model_path)
            save_config(args, best_val_acc, model_path, config_path)
            print(f"  ← new best!")
        else:
            print()

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model  → {model_path}")
    print(f"Best config → {config_path}")


if __name__ == '__main__':
    main()