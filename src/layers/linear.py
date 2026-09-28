"""全結合層（Linear/Dense layer）の実装。"""

import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.x = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.weight + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weight = self.x.T @ grad_output
        self.grad_bias = grad_output.sum(axis=0)
        return grad_output @ self.weight.T
