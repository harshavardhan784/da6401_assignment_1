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
        self.weight_decay = cli_args.weight_decay

        # Build layer-dimension list:
        # [input] -> hidden_1 -> ... -> hidden_n -> [output]
        input_dim = cli_args.input_dim
        output_dim = cli_args.output_dim
        hidden_sizes = cli_args.hidden_size
        num_hidden = cli_args.num_layers

        layer_dims = [input_dim] + hidden_sizes[:num_hidden] + [output_dim]

        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_output = (i == len(layer_dims) - 2)
            activation = "softmax" if is_output else cli_args.activation

            layer = NeuralLayer(
                in_features = layer_dims[i],
                out_features = layer_dims[i + 1],
                activation = activation,
                weight_init = cli_args.weight_init,
                weight_decay = self.weight_decay
            )

            # Tell the output layer whether to use the combined CE+softmax
            # gradient shortcut (avoids double-division and is numerically exact).
            if is_output and self.loss_name == "cross_entropy":
                layer.is_output_ce = True

            self.layers.append(layer)

        # Optimizer is created once and shared across all layers
        self.optimizer = Optimizers.create(
            name = cli_args.optimizer,
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
        """
        dA = compute_loss_gradient(y_true, y_pred, loss_name=self.loss_name)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    # ------------------------------------------------------------------
    def update_weights(self):
        """Apply the optimizer step to every layer."""
        for layer in self.layers:
            self.optimizer.update(layer)

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """Return scalar accuracy. y must be one-hot encoded."""
        y_pred = self.forward(X)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

    # ------------------------------------------------------------------
    def save(self, path):
        """Serialise all weights and biases to a single .npy file."""
        weights = [{"W": layer.W, "b": layer.b} for layer in self.layers]
        np.save(path, weights, allow_pickle=True)
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    def load(self, path):
        """Load weights saved by save() back into the layer list."""
        weights = np.load(path, allow_pickle=True)
        if len(weights) != len(self.layers):
            raise ValueError("Checkpoint layer count does not match current architecture.")
        for layer, wb in zip(self.layers, weights):
            layer.W = wb["W"]
            layer.b = wb["b"]
        print(f"Model loaded from {path}")