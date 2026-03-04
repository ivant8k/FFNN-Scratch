# --------------------------------------------------------------
# Loss Functions
# --------------------------------------------------------------

class Loss:
    # Mean Squared Error: MSE = 1/n * (\sum_{i=1}^{n}(y_{i}-\hat{y})^2
    def mse(y_a, y_p):
        return np.mean((y_a - y_p)**2)
    
    def mse_derivative(y_a, y_p):
        pass

    # Binary Cross-Entropy:
    # \mathcal{L}_{BCE} = -1/n * \sum_{i=1}^{n}(y_{i} log \hat{y} + (1-y_{i}) log (1-\hat{y}))
    def binary_cross_entropy(y_a,y_p):
        pass

    def binary_cross_entropy_derivative(y_a,y_p):
        pass

    # Categorical Cross-Entropy: 
    # \mathcal{L}_{CCE} = -1/n * \sum_{i=1}^{n} \sum_{j=1}^{C} (y_{ij} log \hat{y}_{ij})
    def categorical_cross_entropy(y_a, y_p):
        pass

    def categorical_cross_entropy_derivative(y_a, y_p):
        pass

print("Testing Loss Functions")
