"""Multilayer Perceptron tests module.

This modules contains TestLayer and TestMultilayerPerceptron which tests
behaviour of Multilayer perceptron realted classes.
"""
import numpy as np
import numpy.testing as nptest
import pytest
from neuruspy.multilayer_perceptron import Layer, MultilayerPerceptron
from neuruspy.activation import Relu

def test_layer_predict_h():
    """Tests if predict without activation function works ok.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, -8, -9]])
    w_matrix = np.ones((1, 4))

    layer = Layer((3, 1), Relu, w_matrix)

    nptest.assert_equal(layer.predict_h(data), np.array([[7], [16], [-9]]))

def test_layer_predict():
    """Tests if predict works ok.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, -8, -9]])
    w_matrix = np.ones((1, 4))

    layer = Layer((3, 1), Relu, w_matrix)

    nptest.assert_equal(layer.predict(data), np.array([[7], [16], [0]]))

def test__layer_backpropagate_last_layer():
    """Tests if backpropagation works ok.

    Tests if backpropagation works ok by backpropagating with a layer
    as if it was the last of a network, so it receives as delta the
    difference between output and desired output.
    """
    data = np.array([[1, 2, 3], [7, -8, -9]])
    w_matrix = np.ones((1, 4))
    desired = np.array([[2], [4]])

    layer = Layer((3, 1), Relu, w_matrix)

    expected_w = w_matrix + (2-7) * np.array([1, 1, 2, 3])
    error = desired - layer.predict(data)

    expected_prev_delta = np.array([[-5, -5, -5], [0, 0, 0]])

    prev_delta = layer.backpropagate(data, error, eta=1)

    nptest.assert_equal(layer.w_matrix, expected_w)
    nptest.assert_equal(prev_delta, expected_prev_delta)

def test_layer_w_not_matching_dimensions():
    """Tests if creating a layer with wrong w raises exception.

    Tests if creating a layer with incorrect weight matrix dimensions
    raises an ValueError exception.
    """
    w_matrix = np.ones((1, 3))

    with pytest.raises(ValueError):
        Layer((3, 1), Relu, w_matrix)

def test_perceptron_predict_internal():
    """Tests if predict internal works ok.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    w1_matrix = np.ones((2, 4))
    w2_matrix = np.ones((1, 3)) * 2

    output = [np.array([[7, 16, 25], [7, 16, 25]]).T]
    output.append(np.array([[(7*2+1)*2], [(16*2+1)*2], [(25*2+1)*2]]))

    perceptron = MultilayerPerceptron()
    perceptron.add_layer(Layer((3, 2), Relu, w1_matrix))
    perceptron.add_layer(Layer((2, 1) ,Relu, w2_matrix))

    nptest.assert_equal(output, perceptron.predict_internal(data))

def test_perceptron_predict():
    """Tests if perceptron's predict works ok.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    w1_matrix = np.ones((2, 4))
    w2_matrix = np.ones((1, 3)) * 2

    output = np.array([[(7*2+1)*2], [(16*2+1)*2], [(25*2+1)*2]])

    perceptron = MultilayerPerceptron()
    perceptron.add_layer(Layer((3, 2), Relu, w1_matrix))
    perceptron.add_layer(Layer((2, 1) ,Relu, w2_matrix))

    nptest.assert_equal(output, perceptron.predict(data))

def test_perceptron_evaluate():
    """Tests if perceptron's works ok.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    w1_matrix = np.ones((2, 4))
    w2_matrix = np.ones((1, 3)) * 2

    desired = np.array([[5, 4, 3]]).T

    output = ((30-5)**2 + (66-4)**2 + (102-3)**2) / 3

    perceptron = MultilayerPerceptron()
    perceptron.add_layer(Layer((3, 2), Relu, w1_matrix))
    perceptron.add_layer(Layer((2, 1) ,Relu, w2_matrix))

    assert output == perceptron.evaluate(data, desired)

def test_perceptron_adding_a_not_matching_layer():
    """Tests if adding a not matching layer to a perceptron fails.

    Tests if adding a layer to a perceptron which does not match its inputs
    with last layer outputs raises a ValueError exception.
    """
    perceptron = MultilayerPerceptron()
    perceptron.add_layer(Layer((1, 2), Relu))

    with pytest.raises(ValueError):
        perceptron.add_layer(Layer((3, 1), Relu))

def test_perceptron_train_two_layers():
    """Tests if training a two layers perceptron works ok.
    """
    perceptron = MultilayerPerceptron()
    w1_matrix = np.ones((1,2))
    w2_matrix = np.ones((1,2)) * 2

    data = np.array([[1], [2], [3]])
    desired = np.array([[7], [9], [11]])

    perceptron.add_layer(Layer((1, 1), Relu, w1_matrix))
    perceptron.add_layer(Layer((1, 1), Relu, w2_matrix))

    perceptron.train(data, desired, eta=1, max_iterations=1)

    nptest.assert_equal(perceptron.w_matrixes[0], np.array([[7, 13]]))
    nptest.assert_equal(perceptron.w_matrixes[1], np.array([[5, 11]]))
