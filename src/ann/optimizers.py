"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""
import numpy as np


class BaseOptimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, layer):
        raise NotImplementedError


# ------------------------------------------------------------------
class SGD(BaseOptimizer):
    def update(self, layer):
        layer.W -= self.lr * layer.grad_W
        layer.b -= self.lr * layer.grad_b


# ------------------------------------------------------------------
class Momentum(BaseOptimizer):
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
    Nesterov Accelerated Gradient — true look-ahead implementation.
        W_lookahead = W - lr * beta * v_prev
        grad        = gradient(loss at W_lookahead)
        v           = beta * v + (1 - beta) * grad
        W           = W - lr * v
    Training loop calls apply_lookahead() → forward/backward → restore_weights() → update().
    """
    def __init__(self, learning_rate, beta=0.9):
        super().__init__(learning_rate)
        self.beta    = beta
        self.v_W     = {}
        self.v_b     = {}
        self._W_orig = {}
        self._b_orig = {}

    def apply_lookahead(self, layers):
        for layer in layers:
            lid = id(layer)
            if lid not in self.v_W:
                self.v_W[lid] = np.zeros_like(layer.W)
                self.v_b[lid] = np.zeros_like(layer.b)
            self._W_orig[lid] = layer.W.copy()
            self._b_orig[lid] = layer.b.copy()
            layer.W -= self.lr * self.beta * self.v_W[lid]
            layer.b -= self.lr * self.beta * self.v_b[lid]

    def restore_weights(self, layers):
        for layer in layers:
            lid = id(layer)
            layer.W = self._W_orig[lid]
            layer.b = self._b_orig[lid]

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
class RMSProp(BaseOptimizer):
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
class Optimizers:
    @staticmethod
    def create(name, learning_rate):
        mapping = {
            "sgd":      SGD,
            "momentum": Momentum,
            "nag":      NAG,
            "rmsprop":  RMSProp,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported optimizer: '{name}'. "
                             f"Choose from {list(mapping.keys())}.")
        return mapping[name](learning_rate)