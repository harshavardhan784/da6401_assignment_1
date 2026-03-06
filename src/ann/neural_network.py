"""
Main Neural Network Model

Key contracts with autograder
------------------------------
forward()  → RAW LOGITS (Z of output layer, no softmax).

backward() → Tuple (grad_W_list, grad_b_list) ordered FIRST layer to LAST layer.
             Call convention matches forward():
                 y_pred = model.forward(X)          # raw logits
                 grad_W, grad_b = model.backward(y_true, y_pred)
             backward() applies softmax internally; pass raw logits as y_pred.

             grad_W[0]  → gradient of W for layer 0  (input → hidden1)  FIRST
             grad_W[-1] → gradient of W for output layer                 LAST
             All layer.grad_W / layer.grad_b attributes are also populated.
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

    def _build_layers(self, layer_dims):
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

    def forward(self, X):
        """Returns RAW LOGITS — softmax is NOT applied."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, y_true, y_pred_logits):
        """
        Backpropagate through all layers.

        Parameters
        ----------
        y_true         : one-hot labels  (N, C)
        y_pred_logits  : RAW LOGITS from forward()  (N, C).
                         softmax is applied internally here.

        Returns
        -------
        Tuple (grad_W_list, grad_b_list) ordered FIRST layer to LAST layer:
            grad_W[0]  → first (input→hidden1) layer gradient
            grad_W[-1] → output layer gradient
        All layer.grad_W / layer.grad_b attributes are also populated.
        """
        # Apply softmax internally to convert logits → probabilities
        y_pred_probs = softmax(y_pred_logits)

        # For CE:  combined CE+softmax gradient = probs - y_true (correct dL/dZ_out)
        # For MSE: need softmax Jacobian-vector product to get correct dL/dZ_out
        #          because softmax is applied here (outside the linear output layer)
        if self.loss_name == 'cross_entropy':
            dA = compute_loss_gradient(y_true, y_pred_probs, loss_name='cross_entropy')
        else:
            mse_grad = compute_loss_gradient(y_true, y_pred_probs, loss_name=self.loss_name)
            dA = softmax_jacobian_vector_product(mse_grad, y_pred_probs)

        # Backprop reversed; collect gradients in LAST→FIRST order internally
        grad_W_list = []
        grad_b_list = []
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # Reverse to FIRST→LAST order:
        #   grad_W_list[0]  = first layer (input → hidden1)
        #   grad_W_list[-1] = output layer
        grad_W_list = grad_W_list[::-1]
        grad_b_list = grad_b_list[::-1]

        return grad_W_list, grad_b_list

    def update_weights(self):
        """Apply the optimizer step to every layer."""
        for layer in self.layers:
            self.optimizer.update(layer)

    def evaluate(self, X, y):
        """Return scalar accuracy. y must be one-hot encoded."""
        probs       = softmax(self.forward(X))
        predictions = np.argmax(probs, axis=1)
        true_labels = np.argmax(y,     axis=1)
        return np.mean(predictions == true_labels)

    def save(self, path):
        np.save(path, self.get_weights())
        print(f"Model saved to {path}")

    def load(self, path):
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

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
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