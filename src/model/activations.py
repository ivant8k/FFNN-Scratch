import numpy as np

class Activation:

    # linear
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    # relu
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    # sigmoid
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # tanh
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return (2 /(np.exp(x) + np.exp(-x)))**2

    # softmax
    def softmax(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            z = x - np.max(x)
            exp_x = np.exp(z)
            return exp_x / np.sum(exp_x)

        z = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(z)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(self, x):
        S = self.softmax(x)
        if S.ndim == 1:
            S_vector = S.reshape(-1, 1)
            diag = np.diagflat(S)
            return diag - np.dot(S_vector, S_vector.T)

        # Batch Jacobian: shape (batch, n_class, n_class)
        jacobian = -np.einsum('bi,bj->bij', S, S)
        idx = np.arange(S.shape[1])
        jacobian[:, idx, idx] += S
        return jacobian

    # leaky relu (bonus)
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x>0, 1, alpha)

    # softplus (bonus)
    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def softplus_derivative(self, x):
        return self.sigmoid(x)