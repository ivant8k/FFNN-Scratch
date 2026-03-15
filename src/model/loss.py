import numpy as np

class Loss:
    # mse
    def mse(self, y_a, y_p, epsilon=1e-15):
        return np.mean((y_a - y_p)**2)
    
    def mse_derivative(self, y_a, y_p):
        return -2 * (y_a - y_p) / y_p.shape[0]

    # bce
    def binary_cross_entropy(self, y_a,y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1 - epsilon)
        return -np.mean(y_a * np.log(y_p) + (1-y_a) * np.log(1-y_p))

    def binary_cross_entropy_derivative(self, y_a,y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return (y_p - y_a) / (y_p * (1 - y_p) * y_p.shape[0])

    # cce
    def categorical_cross_entropy(self, y_a, y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return -np.mean(np.sum(y_a * np.log(y_p), axis = 1))

    def categorical_cross_entropy_derivative(self, y_a, y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return -(y_a / (y_p * y_p.shape[0]))