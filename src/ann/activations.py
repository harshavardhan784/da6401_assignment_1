"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax

Softmax backward convention
---------------------------
activation_backward('softmax') now receives (Z, name, A=None).
When A (the cached softmax output) is provided it computes the
correct Jacobian-vector product:

    dL/dZ_i = A_i * (dL/dA_i - sum_j(A_ij * dL/dA_ij))

This is required for MSE loss where the full Jacobian matters.
For cross-entropy the caller (objective_functions.py) already
returns the combined dL/dZ = (y_pred - y_true) directly, so
NeuralLayer passes activation_backward result of ones and the
combined gradient flows through unchanged — same behaviour as before.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    # Clip to avoid overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def softmax(x):
    x = np.atleast_2d(x)
    # Subtract max for numerical stability
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_jacobian_vector_product(dA, A):
    """
    Efficient Jacobian-vector product for softmax (no explicit N x C x C matrix).
    For each sample i:
        dL/dZ_i = A_i * (dA_i - dot(A_i, dA_i))
    Returns array of same shape as dA.
    """
    dot = np.sum(A * dA, axis=1, keepdims=True)   # (N, 1)
    return A * (dA - dot) # (N, C)

def softmax_derivative(x):
    """Full per-sample Jacobian (kept for reference / W&B report analysis)."""
    s = softmax(x)
    batch_size, num_classes = s.shape
    jacobian = np.zeros((batch_size, num_classes, num_classes))
    for i in range(batch_size):
        si = s[i].reshape(-1, 1)
        jacobian[i] = np.diagflat(si) - np.dot(si, si.T)
    return jacobian


# Unified wrappers used by NeuralLayers
def activation_forward(Z, name):
    """Apply named activation to pre-activation matrix Z."""
    if name == "relu":
        return relu(Z)
    elif name == "sigmoid":
        return sigmoid(Z)
    elif name == "tanh":
        return tanh(Z)
    elif name == "softmax":
        return softmax(Z)
    else:
        raise ValueError(f"Unknown activation: {name}")


def activation_backward(Z, name, A=None):
    """
    Compute the derivative contribution for the backward pass.

    Parameters
    ----------
    Z : ndarray  — pre-activation values cached during forward pass
    name : str   — activation name
    A : ndarray  — post-activation (softmax output); required for softmax

    Returns
    -------
    For relu / sigmoid / tanh:
        Element-wise derivative array (same shape as Z).
        NeuralLayer multiplies this by dA element-wise.

    For softmax:
        Returns the Jacobian-vector product dL/dZ directly.
        NeuralLayer must therefore pass in dA (= dL/dA) and use
        the returned value as dZ without further element-wise multiply.
        (NeuralLayer handles this via the 'softmax' special-case flag.)
    """
    if name == "relu":
        return relu_derivative(Z)
    elif name == "sigmoid":
        return sigmoid_derivative(Z)
    elif name == "tanh":
        return tanh_derivative(Z)
    elif name == "softmax":
        # A must be provided; it is cached in NeuralLayer as self.A
        if A is None:
            raise ValueError("activation_backward for 'softmax' requires A (cached output).")
        # Returns dL/dZ directly (not a multiplicative factor)
        # NeuralLayer.backward() will use this as dZ, not dA * this.
        return A   # sentinel — actual JVP computed in NeuralLayer using dA
    else:
        raise ValueError(f"Unknown activation: {name}")