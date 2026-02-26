"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam

BUG FIX — Adam & Nadam timestep:
    self.t was incremented inside update(layer), which is called once
    PER LAYER per batch. With 4 layers, t incremented 4x per batch,
    making bias correction wrong for every single batch.
    Fix: each layer tracks its own t via a per-layer dict self.t_dict[lid].
    This gives each layer the correct step count independently.
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
class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation.

    FIX: self.t is now tracked per-layer (self.t_dict[lid]) so that
    each layer's bias correction uses the correct step count regardless
    of how many layers exist in the network.
    Previously self.t was a single counter incremented once per
    update(layer) call — with N layers it incremented N times per batch,
    making bias correction wrong by a factor of N in the denominator.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t_dict = {}          # per-layer step counter  ← FIX
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def update(self, layer):
        lid = id(layer)

        if lid not in self.m_W:
            self.t_dict[lid] = 0  # ← FIX: initialise per-layer counter
            self.m_W[lid] = np.zeros_like(layer.W)
            self.v_W[lid] = np.zeros_like(layer.W)
            self.m_b[lid] = np.zeros_like(layer.b)
            self.v_b[lid] = np.zeros_like(layer.b)

        self.t_dict[lid] += 1    # ← FIX: increment only this layer's counter
        t = self.t_dict[lid]

        self.m_W[lid] = self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W
        self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b

        self.v_W[lid] = self.beta2 * self.v_W[lid] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (layer.grad_b ** 2)

        m_W_hat = self.m_W[lid] / (1 - self.beta1 ** t)
        v_W_hat = self.v_W[lid] / (1 - self.beta2 ** t)
        m_b_hat = self.m_b[lid] / (1 - self.beta1 ** t)
        v_b_hat = self.v_b[lid] / (1 - self.beta2 ** t)

        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


# ------------------------------------------------------------------
class Nadam(BaseOptimizer):
    """
    Nesterov-accelerated Adam (Dozat 2016).

    Same per-layer t_dict fix as Adam applied here.
    """
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(learning_rate)
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.t_dict = {}          # ← FIX
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def update(self, layer):
        lid = id(layer)

        if lid not in self.m_W:
            self.t_dict[lid] = 0  # ← FIX
            self.m_W[lid] = np.zeros_like(layer.W)
            self.v_W[lid] = np.zeros_like(layer.W)
            self.m_b[lid] = np.zeros_like(layer.b)
            self.v_b[lid] = np.zeros_like(layer.b)

        self.t_dict[lid] += 1    # ← FIX
        t = self.t_dict[lid]

        self.m_W[lid] = self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W
        self.m_b[lid] = self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b

        self.v_W[lid] = self.beta2 * self.v_W[lid] + (1 - self.beta2) * (layer.grad_W ** 2)
        self.v_b[lid] = self.beta2 * self.v_b[lid] + (1 - self.beta2) * (layer.grad_b ** 2)

        v_W_hat = self.v_W[lid] / (1 - self.beta2 ** t)
        v_b_hat = self.v_b[lid] / (1 - self.beta2 ** t)

        # Nesterov: use t+1 bias correction for the momentum term
        m_W_nesterov = (self.beta1 * self.m_W[lid] + (1 - self.beta1) * layer.grad_W) \
                       / (1 - self.beta1 ** (t + 1))
        m_b_nesterov = (self.beta1 * self.m_b[lid] + (1 - self.beta1) * layer.grad_b) \
                       / (1 - self.beta1 ** (t + 1))

        layer.W -= self.lr * m_W_nesterov / (np.sqrt(v_W_hat) + self.eps)
        layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.eps)


# ------------------------------------------------------------------
class Optimizers:
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