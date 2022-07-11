"""Activation tests module.

This module contains TestActivation class which tests activation functions
behaviour.
"""
import numpy as np
import numpy.testing as nptest
from neuruspy import activation

def test_activation_function_does_nothing():
    """Tests if calling base class does nothing.

    Tests if calling back and forward from base abstract class return None.
    """
    assert activation.ActivationFunction.forward(0) is None
    assert activation.ActivationFunction.back(0) is None

def test_relu_forward():
    """Tests if RELU forward works ok.
    """
    data = np.array([-5, 2, 3, -8, 0, 4])
    output = np.array([0, 2, 3, 0, 0, 4])

    nptest.assert_equal(activation.Relu.forward(data), output)

def test_relu_back():
    """Tests if RELU backward works ok.
    """
    data = np.array([-5, 2, 3, -8, 0, 4])
    output = np.array([0, 1, 1, 0, 0, 1])

    nptest.assert_equal(activation.Relu.back(data), output)

def test_tanh_forward():
    """Tests if hyperbolic tangent forward works ok.
    """
    data = np.array([-5, 2, 3, -8, 0, 4])
    output = np.array([-0.99991, 0.96403, 0.99505, -0.99999, 0, 0.99933])

    nptest.assert_almost_equal(activation.Tanh.forward(data), output, 5)

def test_tanh_back():
    """Tests if hyperbolic tangent backward works ok.
    """
    data = np.array([-5, 2, 3, -8, 0, 4])
    output = np.array([1.8158e-4, 0.07065, 9.8660e-3, 4.5014e-7, 1, 1.3401e-3])

    nptest.assert_almost_equal(activation.Tanh.back(data), output, 5)
