# --------------------------------------------------------------
# Loss Functions
# --------------------------------------------------------------
import numpy as np

class Loss:
    # Mean Squared Error: MSE = 1/n * (\sum_{i=1}^{n}(y_{i}-\hat{y})^2
    def mse(self, y_a, y_p, epsilon=1e-15):
        return np.mean((y_a - y_p)**2)
    
    def mse_derivative(self, y_a, y_p):
        return -2 * (y_a - y_p) / y_p.shape[0]

    # Binary Cross-Entropy:
    # \mathcal{L}_{BCE} = -1/n * \sum_{i=1}^{n}(y_{i} log \hat{y} + (1-y_{i}) log (1-\hat{y}))
    def binary_cross_entropy(self, y_a,y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1 - epsilon)
        return -np.mean(y_a * np.log(y_p) + (1-y_a) * np.log(1-y_p))

    def binary_cross_entropy_derivative(self, y_a,y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return (y_p - y_a) / (y_p * (1 - y_p) * y_p.shape[0])

    # Categorical Cross-Entropy: 
    # \mathcal{L}_{CCE} = -1/n * \sum_{i=1}^{n} \sum_{j=1}^{C} (y_{ij} log \hat{y}_{ij})
    def categorical_cross_entropy(self, y_a, y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return -np.mean(np.sum(y_a * np.log(y_p), axis = 1))

    def categorical_cross_entropy_derivative(self, y_a, y_p, epsilon=1e-15):
        y_p = np.clip(y_p, epsilon, 1-epsilon)
        return -(y_a / (y_p * y_p.shape[0]))

# y_actual_arr = np.array([[1, 0, 0], [0,1,0], [0,0,1]])
# y_pred_arr = np.array([[0.8, 0.1, 0.1], [0.2, 0.8, 0.0], [0.3, 0.3, 0.4]])
# print(">>> Testing Loss Functions")
# loss = Loss()
# print("Actual: \n", y_actual_arr)
# print("Predict: \n", y_pred_arr)
# print()
# print("1. MSE")
# print("MSE: ", loss.mse(y_actual_arr, y_pred_arr))
# print("MSE Derivative: ", loss.mse_derivative(y_actual_arr, y_pred_arr))
# print()
# print("2. BCE")
# print("BCE: ", loss.binary_cross_entropy(y_actual_arr, y_pred_arr))
# print("BCE Derivative: ", loss.binary_cross_entropy_derivative(y_actual_arr, y_pred_arr))
# print()
# print("3. CCE")
# print("CCE: ", loss.categorical_cross_entropy(y_actual_arr, y_pred_arr))
# print("CCE Derivative: ", loss.categorical_cross_entropy_derivative(y_actual_arr, y_pred_arr))
# print()
