# --------------------------------------------------------------
# Activation Functions
# --------------------------------------------------------------

import numpy as np

class Activation:

    # Linear: Linear(x) = x
    def linear(self, x):
        return x

    def linear_derivative(self, x):
        return 1

    # ReLU: ReLU(x) = max(0,x)
    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    # Sigmoid: \sigma(x) = 1/(1+e^-x)
    def sigmoid(self, x):
        return 1/ (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Hyperbolic Tangent (tanh): tanh(x)=(e^x - e^-x)/(e^x + e^x)
    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return (2 /(np.exp(x) + np.exp(-x)))**2

    # Softmax: For vector \overrightarrow(x) = (x_1, x_2, ..., x_n) \in \mathbb{R}^n
    # softmax(\overrightarrow(x))_i = (e^(x_i) / \sum_{j=1}{n} e^(x_j))
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def softmax_derivative(self, x):
        S = self.softmax(x)
        S_vector = S.reshape(-1,1)
        diag = np.diagflat(S)
        jacobian = diag - np.dot(S_vector, S_vector.T)

        return jacobian


    # (Bonus) Leaky ReLU: Leaky ReLU(x) = {x | x > 0} or {\alpha x | x <= 0}
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x>0, 1, alpha)

    # (Bonus) SoftPlus: A(x) = ln(1 + e^x)
    def softplus(self, x):
        return np.log(1 + np.exp(x))

    def softplus_derivative(self, x):
        return self.sigmoid(x)



# test_val = 1
# test_arr = np.array([-1.0, 0.0, 1.0, 2.0])
# Activation = Activation()
# print(">>>Testing Fungsi Aktivasi\n")
# print("1. Linear")
# print(f"Linear({test_val}):", Activation.linear(test_val))
# print(f"Linear Derivative({test_val}):", Activation.linear_derivative(test_val))
# print()
# print("2. ReLU")
# print(f"ReLU({test_arr}):", Activation.relu(test_arr))
# print(f"ReLu Derivative({test_arr}):", Activation.relu_derivative(test_arr))
# print()
# print("3. Sigmoid")
# print("Sigmoid(0):", Activation.sigmoid(0))
# print("Sigmoid Derivative(0):", Activation.sigmoid_derivative(0))
# print()
# print("4. Hyperbolic Tangent (tanh)")
# print(f"Tanh({test_val}):", Activation.tanh(test_val))
# print(f"Tanh Derivative({test_val}):", Activation.tanh_derivative(test_val))
# print()
# print("5. Softmax")
# print(f"Softmax({test_arr}):", Activation.softmax(test_arr))
# print(f"Softmax({test_arr}):", Activation.softmax_derivative(test_arr))
# print()
# print("6. Leaky ReLU")
# print(f"Leaky ReLU({test_arr}):", Activation.leaky_relu(test_arr))
# print(f"Leaky ReLU Derivative({test_arr}):", Activation.leaky_relu_derivative(test_arr))
# print()
# print("7. Softplus")
# print(f"Softplus({test_val}):", Activation.softplus(test_val))
# print(f"Softplus Derivative({test_val}):", Activation.softplus_derivative(test_val))
