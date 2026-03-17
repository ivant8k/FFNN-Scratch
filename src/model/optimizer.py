from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

class GradientDescent:
    r"""
    Gradient Descent optimizer + Regularization

    Standard weight update rule:
    W_new = W_old + \Delta{W_old}
    \Delta{W_old} = -\eta * \nabla{E{w_old}}
    where:
    \eta = learning rate (lr)
    \nabla{E{w_old}} = gradient of the error (loss) E with respect to the weight vector old weight

    Regularization: Penalty to the loss
    L1 (Lasso): adds the absolute value of sum of coefficient as penalty -> \lambda * |w|
    L2 (Ridge): adds the square sum of coefficient as penalty -> \lambda * w^2
    """
    def __init__(self, lr=0.01, reg_type=None, lam=0.0):
        """
        reg_type: None | 'l1' | 'l2'
        lambda: regularization strength (very small number e.g. 1e-4 ~ 1e-2)
        """
        self.lr = lr
        self.reg_type = reg_type
        self.lam = lam

    def step(self, layers: list) -> None:
        """
        Update weights and biases for all Linear layers in 'layers'
        """
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ("w", "b", "dw", "db")):
                if layer.dw is None or layer.db is None:
                    continue

                dw = layer.dw

                if self.reg_type == 'l1':
                    dw = dw + self.lam * np.sign(layer.w)
                elif self.reg_type == 'l2':
                    dw = dw + self.lam * layer.w

                layer.w -= self.lr * dw
                layer.b -= self.lr * layer.db

# Adam
class AdaptiveMomentEstimation:
    r'''
    Adaptive Moment Estimation (Adam) optimizer
    Algorithm 2 -> AdaMax: A variant of Adam based on the infinity norm

    Parameters:
    lr   : float  — learning rate (α). Recommended default: 1e-3
    beta1: float  — decay for 1st moment. Recommended default: 0.9
    beta2: float  — decay for 2nd moment. Recommended default: 0.999
    eps  : float  — numerical stability. Recommended default: 1e-8
    reg_type : None | 'l1' | 'l2'
    lam  : float  — regularization strength
    
    '''
    def __init__ (self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, reg_type=None, lam=0.0):
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.reg_type = reg_type
        self.lam = lam

        self.m = {}  # 1st moment
        self.u = {}  # 2nd moment
        self.t = 0   # timestep

    def step (self, layers):
        self.t += 1 
        # lr = self.lr / (1 - self.beta1**self.t)
        lr = self.lr
        for layer in layers:
            if all(hasattr(layer, attr) for attr in ("w", "b", "dw", "db")):
                if layer.dw is None or layer.db is None:
                    continue
                
                # Initialize first and second moment vectors if not exist
                if layer not in self.m:
                    self.m[layer] = {'w': np.zeros_like(layer.w), 'b': np.zeros_like(layer.b)}
                    self.u[layer] = {'w': np.zeros_like(layer.w), 'b': np.zeros_like(layer.b)}

                # Get gardients w.r.t stochastic objective at timestep t
                dw = layer.dw
                db = layer.db

                if self.reg_type == 'l1':
                    dw = dw + self.lam * np.sign(layer.w)
                elif self.reg_type == 'l2':
                    dw = dw + self.lam * layer.w

                # Update biased first moment estimate
                self.m[layer]['w'] = self.beta1 * self.m[layer]['w'] + (1 - self.beta1) * dw
                self.m[layer]['b'] = self.beta1 * self.m[layer]['b'] + (1 - self.beta1) * db

                # Update the exponentially weighted infinity norm
                self.u[layer]['w'] = np.maximum(self.beta2 * self.u[layer]['w'], np.abs(dw))
                self.u[layer]['b'] = np.maximum(self.beta2 * self.u[layer]['b'], np.abs(db))

                # Update parameters
                step_size = lr / (1 - self.beta1**self.t)
                layer.w -= step_size * (self.m[layer]['w'] / (self.u[layer]['w'] + self.eps))
                layer.b -= step_size * (self.m[layer]['b'] / (self.u[layer]['b'] + self.eps))
                
