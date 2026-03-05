"""
Main Neural Network Model
Orchestrates forward pass, backward pass, weight updates, training loop,
and evaluation.

Key contract with autograder
-----------------------------
forward()  → returns RAW LOGITS (Z of output layer), NOT softmax probabilities.
             Softmax is only applied internally during loss computation.
backward() → returns (grad_W, grad_b) of the FIRST layer (input layer gradients),
             which is what the autograder unpacks as:
             grad_W, grad_b = model.backward(y_true, y_pred)
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import compute_loss, compute_loss_gradient
from ann.optimizers import Optimizers
from ann.activations import softmax


class NeuralNetwork:

    def __init__(self, cli_args):
        self.learning_rate = getattr(cli_args, 'learning_rate', 1e-3)
        self.loss_name     = getattr(cli_args, 'loss',          'cross_entropy')
        self.weight_decay  = getattr(cli_args, 'weight_decay',  0.0)

        input_dim    = getattr(cli_args, 'input_dim',   784)
        output_dim   = getattr(cli_args, 'output_dim',  10)
        num_hidden   = getattr(cli_args, 'num_layers',  3)
        hidden_sizes = getattr(cli_args, 'hidden_size', [128] * num_hidden)
        self._activation  = getattr(cli_args, 'activation',  'relu')
        self._weight_init = getattr(cli_args, 'weight_init', 'xavier')
        optimizer    = getattr(cli_args, 'optimizer',   'adam')

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes] * num_hidden
        else:
            hidden_sizes = list(hidden_sizes)

        layer_dims = [input_dim] + hidden_sizes + [output_dim]
        self._build_layers(layer_dims)

        self.optimizer = Optimizers.create(
            name          = optimizer,
            learning_rate = self.learning_rate,
        )

    # ------------------------------------------------------------------
    def _build_layers(self, layer_dims):
        """Build self.layers from a list of dimensions."""
        self.layers = []
        for i in range(len(layer_dims) - 1):
            is_output = (i == len(layer_dims) - 2)
            # Output layer uses NO activation here — we return raw Z (logits).
            # Softmax is applied separately only when needed (loss, evaluate).
            layer_act = 'linear' if is_output else self._activation

            layer = NeuralLayer(
                in_features  = layer_dims[i],
                out_features = layer_dims[i + 1],
                activation   = layer_act,
                weight_init  = self._weight_init,
                weight_decay = self.weight_decay,
            )

            # For cross-entropy backward, output layer needs softmax grad.
            # We pass is_output_ce so NeuralLayer applies softmax inside backward.
            if is_output and self.loss_name == 'cross_entropy':
                layer.is_output_ce = True

            self.layers.append(layer)

    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Forward pass. Returns RAW LOGITS (pre-softmax Z) from the output layer.
        Softmax is NOT applied here — autograder expects logits.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A   # raw logits, shape (N, 10)

    # ------------------------------------------------------------------
    def _forward_with_softmax(self, X):
        """Internal forward that applies softmax for loss/evaluate use."""
        return softmax(self.forward(X))

    # ------------------------------------------------------------------
    def backward(self, y_true, y_pred_logits):
        """
        Backpropagate through all layers.

        y_pred_logits : raw logits from forward() — softmax applied here
                        before computing loss gradient.

        Returns
        -------
        (grad_W, grad_b) of the FIRST layer — this is what the autograder
        unpacks:  grad_W, grad_b = model.backward(y_true, y_pred)
        """
        # Apply softmax to get probabilities for loss gradient
        y_pred_probs = softmax(y_pred_logits)
        dA = compute_loss_gradient(y_true, y_pred_probs, loss_name=self.loss_name)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

        # Return gradients of the first (input) layer
        first_layer = self.layers[0]
        return first_layer.grad_W, first_layer.grad_b

    # ------------------------------------------------------------------
    def update_weights(self):
        """Apply the optimizer step to every layer."""
        for layer in self.layers:
            self.optimizer.update(layer)

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """Return scalar accuracy. y must be one-hot encoded."""
        logits      = self.forward(X)
        probs       = softmax(logits)
        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y, axis=1)
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
        """
        Set weights from a {W0, b0, W1, b1, ...} dict.
        Rebuilds layers if the dict encodes a different architecture.
        """
        n_layers_in_dict = sum(1 for k in weight_dict if k.startswith('W'))

        if n_layers_in_dict != len(self.layers):
            # Infer layer_dims from weight shapes and rebuild
            layer_dims = []
            for i in range(n_layers_in_dict):
                W = weight_dict[f"W{i}"]
                if i == 0:
                    layer_dims.append(W.shape[0])
                layer_dims.append(W.shape[1])
            self._build_layers(layer_dims)

        for i, layer in enumerate(self.layers):
            if f"W{i}" in weight_dict:
                layer.W = weight_dict[f"W{i}"].copy()
            if f"b{i}" in weight_dict:
                layer.b = weight_dict[f"b{i}"].copy()