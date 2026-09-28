import numpy as np

from src.layers.linear import Linear


def test_linear_forward_shape():
    layer = Linear(4, 3)
    x = np.random.randn(2, 4)
    out = layer.forward(x)
    assert out.shape == (2, 3)


def test_linear_backward_shape():
    layer = Linear(4, 3)
    x = np.random.randn(2, 4)
    layer.forward(x)
    grad_output = np.random.randn(2, 3)
    grad_input = layer.backward(grad_output)
    assert grad_input.shape == x.shape
    assert layer.grad_weight.shape == layer.weight.shape
    assert layer.grad_bias.shape == layer.bias.shape
