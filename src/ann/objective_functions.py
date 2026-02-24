"""
Loss / Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)

Gradient convention
-------------------
Both derivative functions return the *un-averaged* gradient
(shape: batch_size x num_classes). The single /batch_size
averaging is performed inside NeuralLayer.backward(), so callers
must NOT pre-divide by N here — doing so would cause the gradient
to be divided by N twice, breaking the 1e-7 tolerance requirement.

  cross_entropy_derivative  →  (y_pred - y_true)          [combined CE+softmax grad w.r.t Z]
  mse_derivative            →  2 * (y_pred - y_true) / C   [grad w.r.t A, scaled so /N in backward gives correct mean]
"""
import numpy as np


# ------------------------------------------------------------------
# Cross-Entropy  (pairs naturally with Softmax output)
# ------------------------------------------------------------------

def cross_entropy_loss(y_true, y_pred):
    """
    Categorical cross-entropy averaged over the batch.
    A small epsilon prevents log(0).
    """
    epsilon = 1e-12
    y_pred  = np.clip(y_pred, epsilon, 1.0 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def cross_entropy_derivative(y_true, y_pred):
    """
    Combined gradient of cross-entropy loss + softmax activation w.r.t.
    the pre-activation Z of the output layer.

    The combined derivative simplifies to (y_pred - y_true).
    We return this WITHOUT dividing by batch_size; NeuralLayer.backward()
    performs the /N averaging when it computes grad_W and grad_b.
    """
    return y_pred - y_true


# ------------------------------------------------------------------
# Mean Squared Error
# ------------------------------------------------------------------

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_derivative(y_true, y_pred):
    """
    Gradient of MSE w.r.t. y_pred.

    L = mean((y_pred - y_true)^2) = sum / (N * C)
    dL/dy_pred = 2 * (y_pred - y_true) / (N * C)

    We return 2 * (y_pred - y_true) / C  (un-divided by N),
    so that NeuralLayer.backward()'s /N gives the correct /N*C average.

    Note: for MSE paired with a softmax output layer, the softmax
    Jacobian is applied properly inside NeuralLayer.backward() via
    activation_backward('softmax').
    """
    C = y_true.shape[1]
    return 2.0 * (y_pred - y_true) / C


# ------------------------------------------------------------------
# Unified dispatch helpers
# ------------------------------------------------------------------

def compute_loss(y_true, y_pred, loss_name="cross_entropy"):
    if loss_name == "cross_entropy":
        return cross_entropy_loss(y_true, y_pred)
    elif loss_name == "mse":
        return mse_loss(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported loss: '{loss_name}'. "
                         "Choose 'cross_entropy' or 'mse'.")


def compute_loss_gradient(y_true, y_pred, loss_name="cross_entropy"):
    if loss_name == "cross_entropy":
        return cross_entropy_derivative(y_true, y_pred)
    elif loss_name == "mse":
        return mse_derivative(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported loss: '{loss_name}'. "
                         "Choose 'cross_entropy' or 'mse'.")