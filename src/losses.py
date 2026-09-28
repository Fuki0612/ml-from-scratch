"""損失関数の実装。"""

import numpy as np


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray):
    diff = y_pred - y_true
    loss = np.mean(diff ** 2)
    grad = 2 * diff / y_true.size
    return loss, grad
