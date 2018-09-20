import tensorflow as tf
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import model as md
import utils as ut
from sympy import Symbol, solve
from sympy.plotting import plot

# mean and standard deviation
layer = 4
x = np.array([[1, 2, 3], [2, 3, 4], [3, 2, 1], [4, 2, 1]])
y = np.array([[2], [3], [5], [3]])


def weight(feature, labels, shape, mu=0, sigma=0.1):
    shape = [feature.shape[1]] + shape + [labels.shape[1]]
    w = []
    for i in range(len(shape) - 1):
        w.append(np.random.normal(mu, sigma, [shape[i] + 1, shape[i + 1]]))
    return w


def bias(values):
    height, width = values.shape
    _bais = np.ones([height, width + 1])
    _bais[:, 1:] = values
    return _bais


def debias(values):
    return values[:, 1:]


def forward(weights, forwards):
    for i in range(len(weights) - 1):
        forwards.append(bias(np.matmul(forwards[i], weights[i])))
    forwards.append(np.matmul(forwards[-1], weights[-1]))
    return forwards


def backward(label, size, forwards, weights, alpha=0.01):
    sigma = [np.subtract(label, forwards[-1])]
    deltas = [np.matmul(np.transpose(forwards[-2]), sigma[0])]

    for i in range(len(weights) - 1, 0, -1):
        gap = debias(np.matmul(sigma[0], np.transpose(weights[i])))
        sigma.insert(0, gap)
        delta = np.matmul(np.transpose(forwards[i - 1]), gap)
        deltas.insert(0, delta)

    return deltas

weights = weight(x, y, [3, 6])

forwards = forward(weights, [bias(x)])

next = backward(y, len(x), forwards, weights)


