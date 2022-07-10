from abc import abstractmethod
import numpy as np

class ActivationFunction:
    @staticmethod
    @abstractmethod
    def forward(x_values):
        pass

    @staticmethod
    @abstractmethod
    def back(x_values):
        pass

class Tanh(ActivationFunction):
    @staticmethod
    def forward(x):
        return np.tanh(x)
    
    @staticmethod
    def back(x):
        return 1 - np.tanh(x)**2

class Relu(ActivationFunction):
    @staticmethod
    def forward(x_values):
        return (x_values > 0) * x_values

    @staticmethod
    def back(x_values):
        return (x_values > 0) * 1