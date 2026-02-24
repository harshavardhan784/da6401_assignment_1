"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation.

Softmax backward
----------------
For the softmax output layer two cases are handled:

1. Cross-entropy loss  (activation='softmax', loss passes combined dL/dZ)
   objective_functions.cross_entropy_derivative returns (y_pred - y_true)
   which is already dL/dZ (combined CE + softmax gradient).
   We use it directly as dZ — no further Jacobian multiplication needed.

2. MSE loss  (activation='softmax', loss passes dL/dA)
   objective_functions.mse_derivative returns dL/dA.
   We apply the softmax Jacobian-vector product to obtain dL/dZ:
       dZ_i = A_i * (dA_i - sum_j(A_ij * dA_ij))

To distinguish the two cases without coupling to the loss name,
we detect cross-entropy's combined gradient by checking whether
dA already sums to approximately zero per row (which the combined
CE+softmax gradient (y_pred - y_true) satisfies for one-hot labels).
The clean approach is to expose an `is_output_ce` flag set by
NeuralNetwork when loss=cross_entropy.  This is set automatically.
"""
import numpy as np
from ann.activations import activation_forward, activation_backward, softmax_jacobian_vector_product


class NeuralLayer:
    """
    A single fully-connected layer with a configurable activation function.
    Stores intermediate values needed for backprop and exposes
    self.grad_W / self.grad_b after every backward() call.
    """

    def __init__(self, in_features, out_features,
                 activation, weight_init="random", weight_decay=0.0):

        self.in_features     = in_features
        self.out_features    = out_features
        self.activation_name = activation
        self.weight_decay    = weight_decay

        # Flag set by NeuralNetwork: True when this is the output softmax layer
        # AND the loss is cross-entropy (combined gradient shortcut applies).
        self.is_output_ce = False

        # ---- Weight initialisation ----
        if weight_init == "random":
            self.W = np.random.randn(in_features, out_features) * 0.01

        elif weight_init == "xavier":
            limit  = np.sqrt(2.0 / (in_features + out_features))
            self.W = np.random.randn(in_features, out_features) * limit

        else:
            raise ValueError(f"Unsupported weight_init: '{weight_init}'. "
                             "Choose 'random' or 'xavier'.")

        self.b = np.zeros((1, out_features))

        # ---- Cache (filled during forward pass) ----
        self.X = None   # input
        self.Z = None   # pre-activation
        self.A = None   # post-activation

        # ---- Gradients (filled during backward pass) ----
        self.grad_W = None
        self.grad_b = None

    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Compute Z = X @ W + b  then  A = activation(Z).
        Cache X, Z, and A for the backward pass.
        """
        self.X = X
        self.Z = np.dot(X, self.W) + self.b
        self.A = activation_forward(self.Z, self.activation_name)
        return self.A

    # ------------------------------------------------------------------
    def backward(self, dA):
        """
        Given dA (gradient of loss w.r.t this layer's output A),
        compute gradients for W and b, and return dX for the previous layer.

        For the softmax output layer:
          - If loss=cross_entropy: dA is already the combined dL/dZ = (y_pred-y_true).
            Use it directly as dZ.
          - If loss=mse: dA is dL/dA. Apply the softmax Jacobian-vector product
            to obtain dL/dZ.
        """
        batch_size = self.X.shape[0]

        # --- Compute dZ ---
        if self.activation_name == "softmax":
            if self.is_output_ce:
                # Combined CE+softmax gradient: dA is already dL/dZ
                dZ = dA
            else:
                # MSE (or other): apply proper softmax Jacobian-vector product
                dZ = softmax_jacobian_vector_product(dA, self.A)
        else:
            # Hidden layer: element-wise activation derivative
            dZ = dA * activation_backward(self.Z, self.activation_name)

        # --- Gradients averaged over the mini-batch ---
        self.grad_W = np.dot(self.X.T, dZ) / batch_size
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / batch_size

        # L2 weight-decay adds lambda * W to the weight gradient
        if self.weight_decay > 0:
            self.grad_W += self.weight_decay * self.W

        # Gradient flowing back to the previous layer
        dX = np.dot(dZ, self.W.T)
        return dX