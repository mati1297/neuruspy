"""Activation Function module.

This module includes activation functions classes.
"""
from abc import abstractmethod
import numpy as np

class ActivationFunction:
    """Abstract class of activation function."""
    @staticmethod
    @abstractmethod
    def forward(x_values):
        """Activation function for foward propagation.
        """

    @staticmethod
    @abstractmethod
    def back(x_values):
        """Derivative of the activation function for backpropagation.
        """

class Tanh(ActivationFunction):
    """Hyperbolic tangent activation function.
    """
    @staticmethod
    def forward(x_values):
        """Activation function for foward propagation.
        """
        return np.tanh(x_values)

    @staticmethod
    def back(x_values):
        """Derivative of the activation function for backpropagation.
        """
        return 1 - np.tanh(x_values)**2

class Relu(ActivationFunction):
    """Rectified linear unit (RELu) activation function.
    """
    @staticmethod
    def forward(x_values):
        """Activation function for foward propagation.
        """
        return (x_values > 0) * x_values

    @staticmethod
    def back(x_values):
        """Derivative of the activation function for backpropagation.
        """
        return (x_values > 0) * 1
        