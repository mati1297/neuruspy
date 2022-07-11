"""Multilayer Perceptron module.

This module includes MultilayerPerceptron and Layer classes.
"""
import numpy as np
from neuruspy.activation import ActivationFunction

class Layer:
    """Fully connected layer implementation.

    Fully connected layer implementation. It allows to create a layer, train 
    it by gradient descent and backpropagation and predict values with it.

    Attributes:
        activation_function (ActivationFunction): Activation function of the 
             layer.
    """
    def __init__(self, n_inputs, n_outputs,
                    activation_function: ActivationFunction,
                    initial_w: np.ndarray=None, w_random_low:float=-1,
                    w_random_high=1):
        """Initialize a Layer instance.

        Initialize a Layer instance. If a initial weight matrix is not 
        provided, it is filled with random numbers from a uniform 
        distribution.

        Args:
            n_inputs (int): Number of inputs.
            n_outputs (int): Number of outputs.
            activation_function (ActivationFunction): Activation function of the
                 layer.
            initial_w (ndarray): Initial values of the weight matrix. If not
                specified, are generated randomly.
            w_random_low (float): Lower value of the uniform distribution for
                random initialization of weight matrix.
            w_random_high (float): Lower value of the uniform distribution for
                random initialization of weight matrix.
        """
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._activation_function = activation_function
        if initial_w is None:
            self._w = np.random.default_rng().uniform(w_random_low,
                                                        w_random_high,
                                                        (n_outputs,
                                                        self.n_inputs+1))
        else:
            if initial_w.shape != (n_outputs, n_inputs+1):
                raise ValueError("W shape does not match with layer inputs and "\
                                    "outputs.")
            self._w = initial_w

    def _add_bias(self, input_data):
        """Add bias column to input data.

        Add bias column to input data if is neccesary.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.

        Returns:
            ndarray: Input data with bias column added.
        """
        if input_data.shape[1] + 1 == self.n_inputs + 1:
            return np.c_[np.ones((input_data.shape[0], 1)), input_data]
        return input_data

    def predict(self, input_data):
        """Predicts an output for the input data.

        Predicts an output for the input data, evaluating the layer

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.

        Returns:
            ndarray: A two dimensional array with samples' output as rows and variables
            as columns.
        """
        return self._activation_function.forward(self.predict_h(input_data))

    def predict_h(self, input_data):
        """Predicts an output for the input data without activation function.

        Predicts an output for the input data without evaluating the 
        activation function.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.

        Returns:
            ndarray: A two dimensional array with samples' output as rows and 
            variables as columns.
        """
        input_data = self._add_bias(input_data)
        input_data = input_data.T
        return (self._w @ input_data).T

    def backpropagate(self, input_data, next_layer_delta, eta):
        """Updates weight matrix and backpropagate error.

        Updates weight matrix and calculates delta for previous layer.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.
            next_layer_delta (ndarray): Delta of the next layer, to 
                backpropagate.
            eta (float): Learning constant.

        Returns:
            ndarray: A two dimensional array delta for previous layer.
        """
        input_data = self._add_bias(input_data)

        delta = self._activation_function.back(self.predict_h(input_data)) \
                    * next_layer_delta

        prev_layer_delta = delta @ self._w
        self._w += eta * delta.T @ input_data

        return prev_layer_delta[:, 1:]

    @property
    def w(self):
        """Weight matrices.

        Returns:
            ndarray: Layer's weight matrix.
        """
        return self._w

    @property
    def n_inputs(self):
        """Number of inputs.

        Returns:
            int: Number of inputs.
        """
        return self._n_inputs

    @property
    def n_outputs(self):
        """Number of outputs.

        Returns:
            int: Number of outputs.
        """
        return self._n_outputs

class MultilayerPerceptron:
    """Multilayer Perceptron implementation.
    
    Multilayer Perceptron implementation. It allows to create a multilayer
    perceptron by chaining different layers (represented by class Layer
    instances), train it and predict new results using it.

    Attributes:
        layers (list): List of Layer instances with the layers of the
            perceptron.
    """
    def __init__(self):
        """Initialize a MultilayerPreceptron instance.

        Initialize a MultilayerPerceptron instance. It takes no arguments.
        """
        self._layers = []

    def add_layer(self, new_layer: Layer):
        """Add a layer to the perceptron.

        Add a layer to the perceptron, chaining it at the end of the network.

        Args:
            new_layer (Layer): Layer to be added to the perceptron. It should have
                as many inputs as outputs of last layer of the network.
        """
        if not self.isempty:
            if self._layers[-1].n_outputs != new_layer.n_inputs:
                raise ValueError("Number of inputs of new layer does not "\
                                    "match number of outputs of last layer.")
        self._layers.append(new_layer)

    def predict(self, input_data: np.ndarray):
        """Predicts an output for the input data.

        Predicts an output for the input data, evaluating the newtork.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.

        Returns:
            ndarray: A two dimensional array with samples' output as rows and variables
            as columns.
        """
        output = input_data
        for layer in self._layers:
            output = layer.predict(output)
        return output

    def predict_internal(self, input_data: np.ndarray):
        """Predicts internal outputs for the input data.

        Predicts outputs of every layer for the input data, evaluating the
        network.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.

        Returns:
            list: A list of two dimensional array with samples' outputs rows 
            and variables as columns for every layer.
        """
        output = [input_data]
        for layer in self._layers:
            output.append(layer.predict(output[-1]))
        return output[1:]

    def evaluate(self, input_data: np.ndarray, 
                    desired_output_data: np.ndarray):
        """Predicts and evaluates output for the input data.

        Predicts an output for the input data, and then computes the mean
        squared error from the desired output.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.
            desired_output_data (ndarray): Two dimensional array with desired
                outputs as rows and variables as columns.

        Returns:
            float: Computed squared mean error for all the samples.
        """
        ecm = np.sum((desired_output_data - self.predict(input_data))**2)
        return ecm / input_data.shape[0]

    def train(self, input_data: np.ndarray, desired_output_data: np.ndarray,
                eta: float=1000, max_iterations: int=1000):
        """Trains the Multilayer Perceptron

        Trains the Multilayer Perceptron using gradient descent and
        backpropagation while the squared mean error for every iteration.

        Args:
            input_data (ndarray): Two dimensional array with samples as rows 
                and variables as columns.
            desired_output_data (ndarray): Two dimensional array with desired 
                outputs as rows and variables as columns.
            eta (float): Learning constant. By default is set 0.01.
            max_iterations (int): Maximum number of iterations of the
                algorithm. By default is set to 1000.

        Returns:
            list: A list of the squared mean error for every iteration.
        """
        error = []

        last_error = self.evaluate(input_data, desired_output_data)
        error.append(last_error)

        iteration = 0
        while last_error > 0 and iteration < max_iterations:
            internal_outputs = self.predict_internal(input_data)
            next_layer_delta = desired_output_data - internal_outputs[-1]

            internal_outputs.insert(0, input_data)
            internal_outputs.pop()

            for i in reversed(range(0, self.n_layers)):
                next_layer_delta = self._layers[i]. \
                                    backpropagate(internal_outputs[i], 
                                                  next_layer_delta, eta)

            last_error = self.evaluate(input_data, desired_output_data)
            error.append(last_error)

            iteration += 1

        return error


    @property
    def n_layers(self):
        """Number of layers of the perceptron.
        
        Returns:
            int: number of layers.
        """
        return len(self._layers)

    @property
    def isempty(self):
        """Asks if the perceptron is empty.

        Returns:
            bool: True if perceptron has no layers in it. False otherwise.
        """
        return not self.n_layers

    @property
    def w(self):
        """List of weight matrices of the layers of the perceptron.

        Returns:
            list: List of ndarrays containing layers' weight matrices.
        """
        w_output = []
        for layer in self._layers:
            w_output.append(layer.w)
        return w_output
