"""

"""

import numpy as np
from neuruspy.activation import ActivationFunction

class MultilayerPerceptron:
    def __init__(self):
        self._layers = []

    def add_layer(self, new_layer):
        if not self.isempty:
            if self._layers[-1].n_outputs != new_layer.n_inputs:
                raise ValueError("Number of inputs of new layer does not match\
                                 number of outputs of last layer.")
        self._layers.append(new_layer)

    def predict(self, input_data):
        output = input_data
        for layer in self._layers:
            output = layer.predict(output)
        return output

    def predict_internal(self, input_data):
        output = [input_data]
        for layer in self._layers:
            output.append(layer.predict(output[-1]))
        return output[1:]

    def evaluate(self, input_data, desired_data):
        ecm = np.sum((desired_data - self.predict(input_data))**2)
        return ecm / input_data.shape[0]

    def train(self, input_data, desired_data, eta=0.01, max_iterations=1000):
        error = []

        last_error = self.evaluate(input_data, desired_data)\
                        * input_data.shape[0] * 0.5
        error.append(last_error)

        iteration = 0
        while last_error > 0 and iteration < max_iterations:
            internal_outputs = self.predict_internal(input_data)
            next_layer_delta = desired_data - internal_outputs[-1]

            internal_outputs.insert(0, input_data)
            internal_outputs.pop()

            for i in reversed(range(0, self.n_layers)):
                next_layer_delta = self._layers[i]. \
                                    backpropagate(internal_outputs[i], 
                                                  next_layer_delta, eta)

            last_error = self.evaluate(input_data, desired_data)
            error.append(last_error)

            iteration += 1

        return error


    @property
    def n_layers(self):
        return len(self._layers)

    @property
    def isempty(self):
        return not self.n_layers

    @property
    def w(self):
        w_output = []
        for layer in self._layers:
            w_output.append(layer.w)
        return w_output

class Layer:
    def __init__(self, n_inputs, n_outputs,
                activation_function: ActivationFunction, initial_w: np.ndarray=None):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._activation_function = activation_function
        # Agregar limites de random
        if initial_w is None:
            self._w = np.random.default_rng().uniform(-1, 1, (n_outputs,\
                        self.n_inputs+1))
        else:
            if initial_w.shape != (n_outputs, n_inputs+1):
                raise ValueError("W shape does not match with layer inputs and \
                                outputs.")
            self._w = initial_w

    def _add_bias(self, input_data):
        if input_data.shape[1] + 1 == self.n_inputs + 1:
            return np.c_[np.ones((input_data.shape[0], 1)), input_data]
        return input_data

    def predict(self, input_data):
        return self._activation_function.forward(self.predict_h(input_data))

    def predict_h(self, input_data):
        input_data = self._add_bias(input_data)
        input_data = input_data.T
        return (self._w @ input_data).T

    def backpropagate(self, input_data, next_layer_delta, eta):
        input_data = self._add_bias(input_data)

        delta = self._activation_function.back(self.predict_h(input_data)) \
                * next_layer_delta

        prev_layer_delta = delta @ self._w
        self._w += eta * delta.T @ input_data

        return prev_layer_delta[:, 1:]

    @property
    def w(self):
        return self._w

    @property
    def n_inputs(self):
        return self._n_inputs

    @property
    def n_outputs(self):
        return self._n_outputs
        