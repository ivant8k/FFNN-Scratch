# --------------------------------------------------------------
# GD & Regularization
# --------------------------------------------------------------
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if src(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np

class GradientDescent:
    """
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
        from model.layers import Linear

        for layer in layers:
            if isinstance(layer, Linear):
                dw = layer.dw

                if self.reg_type == 'l1':
                    dw = dw + self.lam * np.sign(layer.w)
                elif self.reg_type == 'l2':
                    dw = dw + self.lam * layer.w

                layer.w -= self.lr * dw
                layer.b -= self.lr * layer.db

# (Bonus) Adaptive Moment Estimation (Adam) Optimizer
class AdaptiveMomentEstimation:
    def __init__ (self):
        pass

    def step (self):
        pass
