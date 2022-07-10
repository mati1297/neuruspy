import unittest
import numpy as np
import numpy.testing as nptest
from neuruspy.multilayer_perceptron import Layer, MultilayerPerceptron
from neuruspy.activation import Relu

class TestLayer(unittest.TestCase):
    def test_add_bias(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        data_bias = np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]])

        layer = Layer(3, 1, Relu)

        nptest.assert_equal(layer._add_bias(data), data_bias)

    def test_add_bias_does_nothing(self):
        data = np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]])
        data_bias = np.array([[1, 1, 2, 3], [1, 4, 5, 6], [1, 7, 8, 9]])

        layer = Layer(3, 1, Relu)

        nptest.assert_equal(layer._add_bias(data), data_bias)

    def test_relu_forward(self):
        data = np.array([-5, 2, 3, -8, 0, 4])
        output = np.array([0, 2, 3, 0, 0, 4])

        nptest.assert_equal(Relu.forward(data), output)

    def test_relu_back(self):
        data = np.array([-5, 2, 3, -8, 0, 4])
        output = np.array([0, 1, 1, 0, 0, 1])

        nptest.assert_equal(Relu.back(data), output)

    def test_predict_h(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, -8, -9]])
        w = np.ones((1, 4))

        layer = Layer(3, 1, Relu, w)

        nptest.assert_equal(layer.predict_h(data), np.array([[7], [16], [-9]]))

    def test_predict(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, -8, -9]])
        w = np.ones((1, 4))

        layer = Layer(3, 1, Relu, w)

        nptest.assert_equal(layer.predict(data), np.array([[7], [16], [0]]))

    def test_backpropagate_last_layer(self):
        data = np.array([[1, 2, 3], [7, -8, -9]])
        w = np.ones((1, 4))
        desired = np.array([[2], [4]])

        layer = Layer(3, 1, Relu, w)

        expected_w = w + (2-7) * np.array([1, 1, 2, 3])
        error = desired - layer.predict(data)

        expected_prev_delta = np.array([[-5, -5, -5], [0, 0, 0]])

        prev_delta = layer.backpropagate(data, error, eta=1)

        nptest.assert_equal(layer.w, expected_w)
        nptest.assert_equal(prev_delta, expected_prev_delta)
    
    def test_w_not_matching_dimensions(self):
        w = np.ones((1, 3))

        self.assertRaises(ValueError, Layer, 3, 1, Relu, w)

class TestMultilayerPerceptron(unittest.TestCase):
    def test_predict_internal(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        w1 = np.ones((2, 4))
        w2 = np.ones((1, 3)) * 2

        output = [np.array([[7, 16, 25], [7, 16, 25]]).T]
        output.append(np.array([[(7*2+1)*2], [(16*2+1)*2], [(25*2+1)*2]]))

        perceptron = MultilayerPerceptron()
        perceptron.add_layer(Layer(3, 2, Relu, w1))
        perceptron.add_layer(Layer(2, 1 ,Relu, w2))

        nptest.assert_equal(output, perceptron.predict_internal(data))

    def test_predict(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        w1 = np.ones((2, 4))
        w2 = np.ones((1, 3)) * 2

        output = np.array([[(7*2+1)*2], [(16*2+1)*2], [(25*2+1)*2]])

        perceptron = MultilayerPerceptron()
        perceptron.add_layer(Layer(3, 2, Relu, w1))
        perceptron.add_layer(Layer(2, 1 ,Relu, w2))

        nptest.assert_equal(output, perceptron.predict(data))

    def test_evaluate(self):
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        w1 = np.ones((2, 4))
        w2 = np.ones((1, 3)) * 2

        desired = np.array([[5, 4, 3]]).T

        output = ((30-5)**2 + (66-4)**2 + (102-3)**2) / 3

        perceptron = MultilayerPerceptron()
        perceptron.add_layer(Layer(3, 2, Relu, w1))
        perceptron.add_layer(Layer(2, 1 ,Relu, w2))

        self.assertEqual(output, perceptron.evaluate(data, desired))

    def test_adding_a_not_matching_layer(self):
        perceptron = MultilayerPerceptron()
        perceptron.add_layer(Layer(1, 2, Relu))

        self.assertRaises(ValueError, perceptron.add_layer, Layer(3, 1, Relu))

    def test_train_two_layers(self):
        perceptron = MultilayerPerceptron()
        w1 = np.ones((1,2))
        w2 = np.ones((1,2)) * 2

        data = np.array([[1], [2], [3]])
        desired = np.array([[7], [9], [11]])

        perceptron.add_layer(Layer(1, 1, Relu, w1))
        perceptron.add_layer(Layer(1, 1, Relu, w2))

        perceptron.train(data, desired, eta=1, max_iterations=1)

        nptest.assert_equal(perceptron.w[0], np.array([[7, 13]]))
        nptest.assert_equal(perceptron.w[1], np.array([[5, 11]]))

if __name__ == "__main__":
    unittest.main()