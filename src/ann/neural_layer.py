"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation.

Output layer uses activation='linear' so forward() returns raw Z (logits).
Softmax is applied externally by NeuralNetwork when needed.
"""
import numpy as np
from ann.activations import activation_forward, activation_backward


class NeuralLayer:

    def __init__(self, in_features, out_features,
                 activation, weight_init="random", weight_decay=0.0):

        self.in_features     = in_features
        self.out_features    = out_features
        self.activation_name = activation
        self.weight_decay    = weight_decay
        self.is_output_ce    = False   # set by NeuralNetwork for CE output layer

        # Weight initialisation
        if weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01
        elif weight_init == "xavier":
            limit  = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * limit
        else:
            raise ValueError(f"Unsupported weight_init: '{weight_init}'.")

        self.b = np.zeros((1, out_features))

        # Cache (filled during forward pass)
        self.X = None
        self.Z = None
        self.A = None

        # Gradients (filled during backward pass)
        self.grad_W = None
        self.grad_b = None

    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Z = X @ W + b
        A = activation(Z)   — for 'linear', A == Z (identity)
        """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b

        if self.activation_name == 'linear':
            self.A = self.Z          # output layer: return raw logits
        else:
            self.A = activation_forward(self.Z, self.activation_name)

        return self.A

    # ------------------------------------------------------------------
    def backward(self, dA):
        """
        Given dA = dL/dA, compute gradients and return dX.

        For the output layer (activation='linear'):
          - If is_output_ce=True: dA is already the combined CE+softmax
            gradient dL/dZ = (softmax(Z) - y_true), use directly.
          - Otherwise: dA passes through unchanged (linear derivative = 1).
        """
        batch_size = self.X.shape[0]

        if self.activation_name == 'linear':
            # dL/dZ = dA (either CE combined grad or straight-through)
            dZ = dA
        elif self.activation_name == 'softmax':
            # Legacy path: if someone builds with softmax activation
            if self.is_output_ce:
                dZ = dA
            else:
                from ann.activations import softmax_jacobian_vector_product
                dZ = softmax_jacobian_vector_product(dA, self.A)
        else:
            dZ = dA * activation_backward(self.Z, self.activation_name)

        self.grad_W = np.dot(self.X.T, dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        if self.weight_decay > 0:
            self.grad_W += self.weight_decay * self.W

        return np.dot(dZ, self.W.T)