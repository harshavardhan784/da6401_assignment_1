"""
Inference Script — evaluate a trained model on the test set.
model.forward() returns raw logits; softmax applied here for metrics.
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
from ann.objective_functions import compute_loss
from ann.activations import softmax


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str, default='best_model.npy',
                        help='Path to saved model weights (.npy)')

    parser.add_argument('-d',   '--dataset',      type=str,   default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-b',   '--batch_size',   type=int,   default=32)
    parser.add_argument('-nhl', '--num_layers',   type=int,   default=3)
    parser.add_argument('-sz',  '--hidden_size',  type=int,   nargs='+',
                        default=[128, 128, 128])
    parser.add_argument('-a',   '--activation',   type=str,   default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-l',   '--loss',         type=str,   default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-o',   '--optimizer',    type=str,   default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr',  '--learning_rate',type=float, default=1e-3)
    parser.add_argument('-wd',  '--weight_decay', type=float, default=0.0)
    parser.add_argument('-w_i', '--weight_init',  type=str,   default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-w_p', '--wandb_project',type=str,   default=None)

    return parser.parse_args()


def load_model(args):
    args.input_dim  = 784
    args.output_dim = 10
    model = NeuralNetwork(args)
    # Load using the approach specified in assignment instructions (1.2):
    # np.load(...).item() -> dict -> model.set_weights(dict)
    data = np.load(args.model_path, allow_pickle=True)
    if data.ndim == 0:
        weight_dict = data.item()
    else:
        # Legacy list-of-dicts format
        weight_dict = None
    if isinstance(weight_dict, dict):
        model.set_weights(weight_dict)
    else:
        model.load(args.model_path)
    return model


def evaluate_model(model, X_test, y_test, loss_name='cross_entropy'):
    logits = model.forward(X_test)          # raw logits
    probs  = softmax(logits)                # apply softmax for metrics
    loss   = compute_loss(y_test, probs, loss_name=loss_name)

    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    return {
        'logits':    logits,
        'loss':      loss,
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall':    recall_score(y_true,    y_pred, average='macro', zero_division=0),
        'f1':        f1_score(y_true,        y_pred, average='macro', zero_division=0),
    }


def main():
    args = parse_arguments()

    print(f"Loading dataset: {args.dataset}")
    _, _, _, _, X_test, y_test = load_data(args.dataset)

    print(f"Loading model from: {args.model_path}")
    model = load_model(args)

    print("Running evaluation...")
    results = evaluate_model(model, X_test, y_test, loss_name=args.loss)

    print("\n=== Evaluation Results ===")
    print(f"  Loss      : {results['loss']:.4f}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-score  : {results['f1']:.4f}")
    print("==========================\n")

    return results


if __name__ == '__main__':
    main()