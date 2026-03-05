"""
Main Neural Network Model
Orchestrates forward pass, backward pass, weight updates, training loop,
and evaluation.
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import compute_loss, compute_loss_gradient
from ann.optimizers import Optimizers


class NeuralNetwork:
    """
    Fully-connected feed-forward network built from NeuralLayer objects.
    Takes a cli_args Namespace (from argparse or constructed manually).
    All attributes are read with getattr + sensible defaults so the
    autograder can pass a minimal Namespace without crashing.

    hidden_size normalisation rules
    --------------------------------
    - int  (e.g. 128)        → replicate num_layers times: [128,128,128]
    - list (e.g. [128,64])   → use AS-IS; num_layers is ignored
      (the autograder may pass hidden_size=[128] meaning exactly 1 hidden layer)
    """

    def __init__(self, cli_args):
        self.learning_rate = getattr(cli_args, 'learning_rate', 1e-3)
        self.loss_name     = getattr(cli_args, 'loss',          'cross_entropy')
        self.weight_decay  = getattr(cli_args, 'weight_decay',  0.0)

        input_dim    = getattr(cli_args, 'input_dim',   784)
        output_dim   = getattr(cli_args, 'output_dim',  10)
        num_hidden   = getattr(cli_args, 'num_layers',  3)
        hidden_sizes = getattr(cli_args, 'hidden_size', [128] * num_hidden)
        activation   = getattr(cli_args, 'activation',  'relu')
        weight_init  = getattr(cli_args, 'weight_init', 'xavier')
        optimizer    = getattr(cli_args, 'optimizer',   'adam')

        # Normalise hidden_sizes
        # - int  → replicate num_hidden times
        # - list → use exactly as provided (do NOT pad or extend)
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_hidden
        else:
            hidden_sizes = list(hidden_sizes)
            # num_layers is informational only when a list is given;
            # the list itself defines how many hidden layers there are.

        layer_dims = [input_dim] + hidden_sizes + [output_dim]

        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_output = (i == len(layer_dims) - 2)
            layer_act = 'softmax' if is_output else activation

            layer = NeuralLayer(
                in_features  = layer_dims[i],
                out_features = layer_dims[i + 1],
                activation   = layer_act,
                weight_init  = weight_init,
                weight_decay = self.weight_decay,
            )

            if is_output and self.loss_name == 'cross_entropy':
                layer.is_output_ce = True

            self.layers.append(layer)

        self.optimizer = Optimizers.create(
            name          = optimizer,
            learning_rate = self.learning_rate,
        )

    # ------------------------------------------------------------------
    def forward(self, X):
        """Pass input through every layer and return the final output."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # ------------------------------------------------------------------
    def backward(self, y_true, y_pred):
        """
        Backpropagate loss gradient through all layers.
        Returns list of gradients from last layer to first.
        """
        dA    = compute_loss_gradient(y_true, y_pred, loss_name=self.loss_name)
        grads = []
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grads.append(dA)
        return grads

    # ------------------------------------------------------------------
    def update_weights(self):
        """Apply the optimizer step to every layer."""
        for layer in self.layers:
            self.optimizer.update(layer)

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """Return scalar accuracy. y must be one-hot encoded."""
        y_pred      = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y,      axis=1)
        return np.mean(predictions == true_labels)

    # ------------------------------------------------------------------
    def save(self, path):
        """Save weights dict {W0, b0, W1, b1, ...} to a .npy file."""
        np.save(path, self.get_weights())
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    def load(self, path):
        """
        Load weights from a .npy file.
        Handles both new dict format and legacy list-of-dicts format.
        """
        data = np.load(path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()

        if isinstance(data, dict):
            self.set_weights(data)
        else:
            if len(data) != len(self.layers):
                raise ValueError("Checkpoint layer count does not match architecture.")
            for layer, wb in zip(self.layers, data):
                layer.W = wb["W"]
                layer.b = wb["b"]

        print(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    def get_weights(self):
        """Return {W0, b0, W1, b1, ...} dict of all layer weights."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    # ------------------------------------------------------------------
    def set_weights(self, weight_dict):
        """Set layer weights from a {W0, b0, W1, b1, ...} dict."""
        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()