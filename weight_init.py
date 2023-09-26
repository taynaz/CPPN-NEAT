import numpy as np
import random
from operator import methodcaller


class WeightInitializer:
    @staticmethod
    def randomNormal(mu=0, sigma=1):
        return np.random.normal(mu, sigma, int(1))  # mean, standard deviation and shape of output numpy array

    @staticmethod
    def randomUniform(min=-0.5, max=2.5):
        return np.random.uniform(min, max)

    @staticmethod
    def ones():
        return 1

    @staticmethod
    def zeros():
        return 0

    @staticmethod
    def get_weight(weight_type):
        return methodcaller(weight_type)
