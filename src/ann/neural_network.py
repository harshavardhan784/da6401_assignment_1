"""
Main Neural Network Model

Key contracts with autograder
------------------------------
forward()  → RAW LOGITS (Z of output layer, no softmax).

backward() → list of (grad_W, grad_b) tuples, ordered from FIRST layer
             to LAST layer.
             Call convention:
                 probs = softmax(model.forward(X))
                 grad_W, grad_b = model.backward(y_true, probs)
             y_pred argument must be PROBABILITIES (already softmaxed).
             All layer.grad_W / layer.grad_b attributes are also
             populated for direct inspection.
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.objective_functions import compute_loss, compute_loss_gradient
from ann.optimizers import Optimizers
from ann.activations import softmax, softmax_jacobian_vector_product


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
            layer_act = 'linear' if is_output else self._activation

            layer = NeuralLayer(
                in_features  = layer_dims[i],
                out_features = layer_dims[i + 1],
                activation   = layer_act,
                weight_init  = self._weight_init,
                weight_decay = self.weight_decay,
            )

            if is_output and self.loss_name == 'cross_entropy':
                layer.is_output_ce = True

            self.layers.append(layer)

    # ------------------------------------------------------------------
    def forward(self, X):
        """Returns RAW LOGITS — softmax is NOT applied."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    # ------------------------------------------------------------------
    def backward(self, y_true, y_pred_probs):
        """
        Backpropagate through all layers.

        Parameters
        ----------
        y_true       : one-hot labels  (N, C)
        y_pred_probs : SOFTMAX PROBABILITIES from forward pass (N, C).
                       Call softmax(model.forward(X)) before passing here.

        Returns
        -------
        Tuple (grad_W_list, grad_b_list) where each list is ordered from
        FIRST layer to LAST layer:
            grad_W[0]  → weights of layer 0  (input → hidden1)
            grad_W[-1] → weights of output layer
        All layer.grad_W / layer.grad_b attributes are also populated.
        """
        # Compute dL/d(probs) — the loss gradient w.r.t. softmax output
        dL_dprobs = compute_loss_gradient(
            y_true, y_pred_probs, loss_name=self.loss_name
        )

        # For CE: dL/dZ_out = dL/dprobs (combined CE+softmax gradient already correct)
        # For MSE: dL/dZ_out = softmax_jacobian @ dL/dprobs  (chain rule via Jacobian)
        if self.loss_name == 'cross_entropy':
            dA = dL_dprobs          # already the correct dL/dZ for output layer
        else:
            # Apply the softmax Jacobian-vector product to get dL/dZ_out
            # This correctly propagates gradients through the external softmax
            # that is NOT part of the output layer's own forward pass.
            dA = softmax_jacobian_vector_product(dL_dprobs, y_pred_probs)

        # Backprop through layers (last → first)
        grad_W_list_reversed = []
        grad_b_list_reversed = []
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list_reversed.append(layer.grad_W)
            grad_b_list_reversed.append(layer.grad_b)

        # Reverse to return FIRST→LAST order
        grad_W_list = grad_W_list_reversed[::-1]
        grad_b_list = grad_b_list_reversed[::-1]

        return grad_W_list, grad_b_list

    # ------------------------------------------------------------------
    def update_weights(self):
        """Apply the optimizer step to every layer."""
        for layer in self.layers:
            self.optimizer.update(layer)

    # ------------------------------------------------------------------
    def evaluate(self, X, y):
        """Return scalar accuracy. y must be one-hot encoded."""
        probs       = softmax(self.forward(X))
        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y,     axis=1)
        return np.mean(predictions == true_labels)

    # ------------------------------------------------------------------
    def save(self, path):
        """Save weights dict {W0, b0, W1, b1, ...} to a .npy file."""
        np.save(path, self.get_weights())
        print(f"Model saved to {path}")

    # ------------------------------------------------------------------
    def load(self, path):
        """Load weights. Handles dict format and legacy list-of-dicts."""
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