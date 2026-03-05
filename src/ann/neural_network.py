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
    Architecture is determined through arguments from CLI.
    """

    def __init__(self, cli_args):
        self.learning_rate = cli_args.learning_rate
        self.loss_name = cli_args.loss
        self.weight_decay = getattr(cli_args, 'weight_decay', 0.0)

        # Build layer-dimension list:
        # [input] -> hidden_1 -> ... -> hidden_n -> [output]
        # Use getattr with defaults so autograder Namespaces without
        # input_dim / output_dim still work.
        input_dim  = getattr(cli_args, "input_dim",  784)
        output_dim = getattr(cli_args, "output_dim", 10)
        hidden_sizes = cli_args.hidden_size
        num_hidden   = cli_args.num_layers

        # Guard: if user passes fewer -sz values than -nhl, raise clearly
        if len(hidden_sizes) < num_hidden:
            raise ValueError(
                f"--num_layers (-nhl) is {num_hidden} but only {len(hidden_sizes)} "
                f"values given for --hidden_size (-sz). "
                f"Provide exactly {num_hidden} values, e.g. -sz " +
                " ".join(["128"] * num_hidden)
            )
        layer_dims = [input_dim] + hidden_sizes[:num_hidden] + [output_dim]

        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_output  = (i == len(layer_dims) - 2)
            activation = "softmax" if is_output else cli_args.activation

            layer = NeuralLayer(
                in_features  = layer_dims[i],
                out_features = layer_dims[i + 1],
                activation   = activation,
                weight_init  = getattr(cli_args, 'weight_init', 'xavier'),
                weight_decay = self.weight_decay
            )

            # Tell the output layer whether to use the combined CE+softmax
            # gradient shortcut (avoids double-division and is numerically exact).
            if is_output and self.loss_name == "cross_entropy":
                layer.is_output_ce = True

            self.layers.append(layer)

        # Optimizer is created once and shared across all layers
        self.optimizer = Optimizers.create(
            name          = getattr(cli_args, 'optimizer', 'adam'),
            learning_rate = self.learning_rate
        )

    # ------------------------------------------------------------------
    def forward(self, X):
        """Pass input forward through every layer and return final logits."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # ------------------------------------------------------------------
    def backward(self, y_true, y_pred):
        """
        Compute gradients via backpropagation.
        Starts with the loss gradient and propagates backwards
        through every layer.
        Returns list of dA values from last layer to first (as required).
        """
        dA = compute_loss_gradient(y_true, y_pred, loss_name=self.loss_name)
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
        y_pred       = self.forward(X)
        predictions  = np.argmax(y_pred, axis=1)
        true_labels  = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

    # ------------------------------------------------------------------
    def save(self, path):
        """
        Serialise weights via get_weights() to a single .npy file.
        Format: dict {W0, b0, W1, b1, ...}
        """
        best_weights = self.get_weights()
        np.save(path, best_weights)
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    def load(self, path):
        """
        Load weights saved by save().
        Supports both the new dict format and the old list-of-dicts format.
        """
        data = np.load(path, allow_pickle=True)

        # np.save on a plain dict wraps it in a 0-d object array
        if data.ndim == 0:
            data = data.item()

        if isinstance(data, dict):
            # New format produced by get_weights() / save()
            self.set_weights(data)
        else:
            # Legacy format: array of {"W": ..., "b": ...} dicts
            if len(data) != len(self.layers):
                raise ValueError(
                    "Checkpoint layer count does not match current architecture."
                )
            for layer, wb in zip(self.layers, data):
                layer.W = wb["W"]
                layer.b = wb["b"]

        print(f"Model loaded from {path}")

    # ------------------------------------------------------------------
    def get_weights(self):
        """Return a flat dict {W0, b0, W1, b1, ...} of all layer weights."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    # ------------------------------------------------------------------
    def set_weights(self, weight_dict):
        """Load weights from a flat dict {W0, b0, W1, b1, ...}."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()