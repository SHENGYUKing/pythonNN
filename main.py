# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_dataset(n_samples=1000, noise=0.1):
    """
    generate dataset
    :param n_samples: int, total number of generated examples
    :param noise: float, noise of dataset
    :return:
        X, np.array, features, size (number of examples, dim of features)
        y, np.array, labels, size (number of examples, 1)
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y


def sigmoid(Z):
    """
    compute the sigmoid function of Z
    :param Z: np.array
    :return:
        A, np.array, activation by sigmoid
        cache, save z for backward propagate
    """
    A = 1. / (1. + np.exp(-1. * Z))
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """
    backward propagation for a single sigmoid unit
    :param dA: np.array, activation gradient
    :param cache: stored Z for backward propagation
    :return:
        dZ, np.array, gradient of the cost with respect to Z
    """
    Z = cache
    s = 1. / (1. + np.exp(-1. * Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu(Z):
    """
    compute the Rectified Linear Unit (ReLU) function of Z
    :param Z: np.array
    :return:
        A, np.array, activation by ReLU
        cache, save z for backward propagate
    """
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    backward propagation for a single ReLU unit
    :param dA: np.array, activation gradient
    :param cache: stored Z for backward propagation
    :return:
        dZ, np.array, gradient of the cost with respect to Z
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert dZ.shape == Z.shape
    return dZ


def nn_init_params(layer_dims):
    """
    initialize parameters of NN
    :param layer_dims: list, dimension of each layer in NN
    :return:
        params, dict, each layer's parameters
    """
    params = {}
    layer_num = len(layer_dims)
    for n in range(1, layer_num):
        # use Xavier initialization
        params['W' + str(n)] = np.random.randn(layer_dims[n], layer_dims[n - 1]) / np.sqrt(layer_dims[n - 1])
        assert params['W' + str(n)].shape == (layer_dims[n], layer_dims[n - 1])
        params['b' + str(n)] = np.zeros((layer_dims[n], 1))
        assert params['b' + str(n)].shape == (layer_dims[n], 1)
    return params


def linear_forward(A, W, b):
    """
    linear unit of a layer's forward propagation
    :param A: np.array, activation from previous layer
    :param W: np.array, weights matrix of current layer
    :param b: np.array, biases vector of current layer
    :return:
        Z, np.array, pre-activation for current activation
        cache, dict, store A, W, b for backward propagate
    """
    Z = W.dot(A) + b
    assert Z.shape == (W.shape[0], A.shape[1])
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    activation of forward propagation
    :param A_prev: np.array, activations from previous layer
    :param W: np.array, weights matrix of current layer
    :param b: np.array, biases vector of current layer
    :param activation: str, activation mode of current layer, "sigmoid" or "relu" (default)
    :return:
        A, np.array, the output of current layer's activation function
        cache, dict, store the linear and activation params and vars
    """
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    else:
        print("Unknown Activation Mode, Use default \"ReLU\"")
        A, activation_cache = relu(Z)
    assert A.shape == (W.shape[0], A_prev.shape[1])
    cache = (linear_cache, activation_cache)
    return A, cache


def nn_forward(X, params):
    """
    forward propagation of NN
    :param X: np.array, input X of dataset
    :param params: dict, output of nn_init_params()
    :return:
        AL, np.array, last post-propagation value
        caches, list, including linear_cache and activation_cache
    """
    caches = []
    A = X
    num_layer = len(params) // 2

    # L-1 layers of LINEAR->RELU
    for n in range(1, num_layer):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, params['W' + str(n)], params['b' + str(n)], 'relu')
        caches.append(cache)
    # Lth layer of LINEAR->SIGMOID
    AL, cache = linear_activation_forward(A, params['W' + str(num_layer)], params['b' + str(num_layer)], 'sigmoid')
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches


def nn_cost(AL, y):
    """
    the cost function of NN
    :param AL: np.array, probability vector corresponding to the labels
    :param y: np.array, true labels, size (1, number of examples)
    :return:
        cost, float, cross-entropy cost
    """
    m = y.shape[1]
    cost = (1./m) * (-1. * np.dot(y, np.log(AL).T) - np.dot(1 - y, np.log(1 - AL).T))
    cost = np.squeeze(cost)

    assert cost.shape == ()
    return cost


def linear_backward(dZ, cache):
    """
    linear unit of a layer's backward propagation
    :param dZ: np.array, gradient of the cost with respect to the current layer's output
    :param cache: tuple of (A_prev, W, b) from the forward propagation in the current layer
    :return:
        dA_prev, np.array, gradient of the cost with respect to the previous layer's activation
        dW, np.array, gradient of the cost with respect to current layer's W
        db, np.array, gradient of the cost with respect to current layer's b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    assert dW.shape == W.shape
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    assert db.shape == b.shape
    dA_prev = np.dot(W.T, dZ)
    assert dA_prev.shape == A_prev.shape

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    backward propagation for current layer
    :param dA: np.array, post-propagation gradient for the current layer
    :param cache: tuple of (linear_cache, activation_cache), stored for computing backward propagation efficiently
    :param activation: str, activation mode of current layer, "sigmoid" or "relu" (default)
    :return:
        dA_prev, np.array, gradient of the cost with respect to the previous layer's activation
        dW, np.array, gradient of the cost with respect to current layer's W
        db, np.array, gradient of the cost with respect to current layer's b
    """
    linear_cache, activation_cache = cache

    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    else:
        print("Unknown Activation Mode, Use default \"ReLU\"")
        dZ = relu_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def nn_backward(AL, y, caches):
    """
    backward propagation of NN
    :param AL: np.array, probability vector of the forward propagation from nn_forward()
    :param y: np.array, true labels vector
    :param caches: list, every cache of linear_activation_forward()
    :return:
        grads, dict, dictionary with the gradients
        grads.dA(n), np.array, gradient of the cost with respect to the nth layer's activation
        grads.dW(n), np.array, gradient of the cost with respect to the nth layer's weights
        grads.db(n), np.array, gradient of the cost with respect to the nth layer's biases
    """
    grads = {}
    num_layer = len(caches)
    m = AL.shape[1]
    y = y.reshape(AL.shape)

    dAL = -1. * (np.divide(y, AL) - np.divide(1 - y, 1 - AL))
    # Lth layer of LINEAR->SIGMOID
    current_cache = caches[num_layer - 1]
    grads["dA" + str(num_layer - 1)], grads["dW" + str(num_layer)], grads["db" + str(num_layer)] = \
        linear_activation_backward(dAL, current_cache, "sigmoid")
    # L-1 layers of LINEAR->RELU
    for n in reversed(range(num_layer - 1)):
        current_cache = caches[n]
        dA_prev_tmp, dW_tmp, db_tmp = linear_activation_backward(grads["dA" + str(n + 1)], current_cache, "relu")
        grads["dA" + str(n)] = dA_prev_tmp
        grads["dW" + str(n + 1)] = dW_tmp
        grads["db" + str(n + 1)] = db_tmp

    return grads


def update_params(params, grads, learning_rate):
    """
    update parameters using gradient descent
    :param params: dict, containing user parameters of NN
    :param grads: dict, cotaining gradients from nn_backward()
    :param learning_rate: float, learning rate of model
    :return:
        params, dict, containing updated parameters
    """
    num_layer = len(params) // 2
    for n in range(num_layer):
        params["W" + str(n + 1)] = params["W" + str(n + 1)] - learning_rate * grads["dW" + str(n + 1)]
        params["b" + str(n + 1)] = params["b" + str(n + 1)] - learning_rate * grads["db" + str(n + 1)]
    return params


def model(X, y, layers_dims, learning_rate=0.001, num_iterations=3000, show=False):
    """
    training model by NN
    :param X: np.array, input dataset X, size (dim of features, number of examples)
    :param y: np.array, true labels of input X, size (1, number of examples)
    :param layers_dims: list, dimension of each layer in NN
    :param learning_rate: float, learning rate of gradient descent
    :param num_iterations: int, number of training iterations
    :param show: bool, True to print cost every 100 iter, default False
    :return:
    """
    costs = []
    params = nn_init_params(layers_dims)

    for i in range(num_iterations):
        # forward propagation
        AL, caches = nn_forward(X, params)
        # compute cost
        cost = nn_cost(AL, y)
        # backward propagation
        grads = nn_backward(AL, y, caches)
        # update parameters
        params = update_params(params, grads, learning_rate)
        if show and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return params, costs


def predict(X, y, params):
    """
    predict the results of NN
    :param X: np.array, input data, size (dim of features, number of examples)
    :param y: np.array, true labels of the input X
    :param params: dict, parameters of the trained model
    :return:
        p, np.array, predictions for the given dataset
    """
    m = X.shape[1]
    n = len(params) // 2
    p = np.zeros((1, m))

    probas, caches = nn_forward(X, params)
    for i in range(0, probas.shape[1]):
        p[0, i] = 1 if probas[0, i] > 0.5 else 0

    print("Accuracy:", str(np.sum((p == y) / m)))

    return p


def plot_cost(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def main():
    np.random.seed(2021)
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    # reshape examples for matrix computing efficiently
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    layer_dims = [2, 4, 8, 4, 1]
    lr = 0.002
    num_iter = 3000
    params, costs = model(X_train, y_train, layer_dims, learning_rate=lr, num_iterations=num_iter, show=True)
    y_pred = predict(X_test, y_test, params)
    plot_cost(costs, lr)


if __name__ == '__main__':
    main()
