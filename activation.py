import enum
import random

import numpy as np
import math
from operator import methodcaller


class Activation:

    def sigmoid(arr):

        arr[-arr > np.log(np.finfo(arr.dtype).max)] = 0.0
        arr = np.exp(-arr)
        arr = 1 / (1.0 + arr)
        return arr

    # Return the arc tangent of x, in radians
    def tanh(x):
        return np.tanh(x)

    # Return the sine of x radians.
    def sin(x):
        try:
            x = np.sin(x)
        except Warning:
            print('x', x)
            x = 1
        return x

    def cosine(x):
        try:
            x = np.cos(x)
        except Warning:
            x = np.ones([1])
        return x

    def gaussian(x):
        x = np.maximum(-3.4, np.minimum(3.4, x))
        return np.exp(-5.0 * x ** 2)


    def relu(x):
        return np.maximum(0, x)

    def leakyrelu(x):
        return np.maximum(0.01 * x, x)

    def inverse(x):
        try:
            z = 1 / x
        except Warning:
            z = 0
        return z

    def absolute(x):
        return abs(x)

    def identity(x):
        return x



    # edit later
    def elu(x):
        return x if x > 0.0 else np.exp(x) - 1

    def softplus(x):
        z = np.maximum(-60.0, np.minimum(60.0, 5.0 * x))
        return 0.2 * np.log(1 + np.exp(z))

    def sawtooth(x):
        return x - np.floor(x)

    def square(x):
        try:
            x = x ** 2
        except Warning:
            print('Sqaure warning')
            x = np.round(x, 4)
            x = x ** 2

        return x

    #
    # def cube(x):
    #     return x ** 3


    def selu(x):
        lam = 1.0507009873554804934193349852946
        alpha = 1.6732632423543772848170429916717
        return lam * x if x > 0.0 else lam * alpha * (np.exp(x) - 1)

    def clamped(x):
        return np.maximum(-1.0, np.minimum(1.0, x))

    def log(x):
        x = np.maximum(1e-7, x)
        return np.log(x)

    def addition(x, z):
        return x + z

    def multiplication(x, z):
        return x * z

    def computeAF(x, activation_name):
        return methodcaller(activation_name, x)


