"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""
import numpy as np


class BaseOptimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, layer):
        raise NotImplementedError


# ------------------------------------------------------------------
class SGD(BaseOptimizer):
    """Vanilla stochastic gradient descent."""

    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


# ------------------------------------------------------------------
class Momentum(BaseOptimizer):
    """SGD with exponential moving average of gradients (heavy ball)."""

    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W  = {}
        self.v_b  = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.v_W:
            self.v_W[lid] = np.zeros_like(layer.W)
            self.v_b[lid] = np.zeros_like(layer.b)

        self.v_W[lid] = self.beta * self.v_W[lid] + (1 - self.beta) * layer.grad_W
        self.v_b[lid] = self.beta * self.v_b[lid] + (1 - self.beta) * layer.grad_b

        layer.W -= self.lr * self.v_W[lid]
        layer.b -= self.lr * self.v_b[lid]


# ------------------------------------------------------------------
class NAG(BaseOptimizer):
    """
    Nesterov Accelerated Gradient.
    The look-ahead step is approximated by adding the current
    momentum term on top of the raw gradient.
    """

    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta = beta
        self.v_W  = {}
        self.v_b  = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.v_W:
            self.v_W[lid] = np.zeros_like(layer.W)
            self.v_b[lid] = np.zeros_like(layer.b)

        self.v_W[lid] = self.beta * self.v_W[lid] + (1 - self.beta) * layer.grad_W
        self.v_b[lid] = self.beta * self.v_b[lid] + (1 - self.beta) * layer.grad_b

        # Nesterov look-ahead update
        layer.W -= self.lr * (self.beta * self.v_W[lid] + layer.grad_W)
        layer.b -= self.lr * (self.beta * self.v_b[lid] + layer.grad_b)


# ------------------------------------------------------------------
class RMSProp(BaseOptimizer):
    """Divides the learning rate by an exponential moving average of squared gradients."""

    def __init__(self, learning_rate, beta=0.9, eps=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps  = eps
        self.s_W  = {}
        self.s_b  = {}

    def update(self, layer):
        lid = id(layer)
        if lid not in self.s_W:
            self.s_W[lid] = np.zeros_like(layer.W)
            self.s_b[lid] = np.zeros_like(layer.b)

        self.s_W[lid] = self.beta * self.s_W[lid] + (1 - self.beta) * (layer.grad_W ** 2)
        self.s_b[lid] = self.beta * self.s_b[lid] + (1 - self.beta) * (layer.grad_b ** 2)

        layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s_W[lid]) + self.eps)
        layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s_b[lid]) + self.eps)


# ------------------------------------------------------------------
class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation.
    Combines momentum (1st moment) and RMSProp (2nd moment)
    with bias-correction in the early steps.
    """

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def update(self, layer):
        self.t += 1
        lid = id(layer)

        if lid not in self.m_W:
            self.m_W[lid] = np.zeros_like(layer.W)
            self.v_W[lid] = np.zeros_like(layer.W)
            self.m_b[lid] = np.zeros_like(layer.b)
            self.v_b[lid] = np.zeros_like(layer.b)

        # First moment (mean)
        self.m_W[lid] = self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W
        self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b

        # Second moment (uncentred variance)
        self.v_W[lid] = self.beta2 * self.v_W[lid] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (layer.grad_b ** 2)

        # Bias correction (important in early iterations when t is small)
        m_W_hat = self.m_W[lid] / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W[lid] / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b[lid] / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b[lid] / (1 - self.beta2 ** self.t)

        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


# ------------------------------------------------------------------
class Nadam(BaseOptimizer):
    """
    Nesterov-accelerated Adam.
    Like Adam but uses the Nesterov look-ahead momentum in the update step.
    """

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def update(self, layer):
        self.t += 1
        lid = id(layer)

        if lid not in self.m_W:
            self.m_W[lid] = np.zeros_like(layer.W)
            self.v_W[lid] = np.zeros_like(layer.W)
            self.m_b[lid] = np.zeros_like(layer.b)
            self.v_b[lid] = np.zeros_like(layer.b)

        # First moment
        self.m_W[lid] = self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W
        self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b

        # Second moment
        self.v_W[lid] = self.beta2 * self.v_W[lid] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (layer.grad_b ** 2)

        # Bias-corrected second moment
        v_W_hat = self.v_W[lid] / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b[lid] / (1 - self.beta2 ** self.t)

        # Nesterov: use the *next* momentum estimate instead of the current one
        m_W_nesterov = (self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W) \
                       / (1 - self.beta1 ** self.t)
        m_b_nesterov = (self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b) \
                       / (1 - self.beta1 ** self.t)

        layer.W -= self.lr * m_W_nesterov / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.eps)


# ------------------------------------------------------------------
class Optimizers:
    """Factory class — call Optimizers.create(name, lr) to get an optimizer."""

    @staticmethod
    def create(name, learning_rate):
        mapping = {
            "sgd":      SGD,
            "momentum": Momentum,
            "nag":      NAG,
            "rmsprop":  RMSProp,
            "adam":     Adam,
            "nadam":    Nadam,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported optimizer: '{name}'. "
                             f"Choose from {list(mapping.keys())}.")
        return mapping[name](learning_rate)