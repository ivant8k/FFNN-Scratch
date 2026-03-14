from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from model.activations import Activation
from model.loss import Loss
from model.optimizer import GradientDescent

class Linear:
    def __init__(self, in_features: int, out_features: int):
        # w - weight matrix, b - bias vector
        self.w = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.dw = None
        self.db = None
    
    def forward(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.cache = x
        return x @ self.w + self.b
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout = np.asarray(dout, dtype=np.float64)
        self.dw = self.cache.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.w.T

class ActivationLayer:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.act = Activation()
        self.cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache = x
        fn = getattr(self.act, self.name)
        return fn(x, **self.kwargs)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        deriv_fn = getattr(self.act, f"{self.name}_derivative")
        return dout * deriv_fn(self.cache, **self.kwargs)
    

# loss wrapper
class LossLayer:
    """
    Loss layer for computing various loss functions.
    ex:
        loss_layer = LossLayer('bce')
    """
    loss_map = {
        'bce': 'binary_cross_entropy',
        'cce': 'categorical_cross_entropy',
        'mse': 'mse'
    }
    def __init__(self, name: str):
        if name not in self.loss_map:
            raise ValueError(f"[LossLayer] loss '{name}' unrecognized"
                             f"Valid options: {list(self.loss_map.keys())}")
        
        self.loss_name = name
        self.loss = Loss()
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        fn = getattr(self.loss, self.loss_map[self.loss_name])
        return float(fn(y_true, y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        deriv_fn = getattr(self.loss, f"{self.loss_map[self.loss_name]}_derivative")
        return deriv_fn(y_true, y_pred)
    
class FFNN:
    def __init__(
            self,
            input_dim: int,
            hidden_dim: list,
            output_dim: int = 1,
            hidden_activation: str = 'sigmoid',
            output_activation: str = 'relu',
            loss_name: str = 'bce',
            act_kwargs: dict = None, # for leaky realu
    ):
        self.layers = []
        act_kwargs = act_kwargs or {}

        dims = [input_dim] + hidden_dim

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            self.layers.append(ActivationLayer(hidden_activation, **act_kwargs))
        
        self.layers.append(Linear(dims[-1], output_dim))
        self.layers.append(ActivationLayer(output_activation))

        self.loss_layer = LossLayer(loss_name)
        print(f"[FFNN] architecture initialized with: {input_dim} -> "
              + f" -> ".join(str(d) for d in hidden_dim)
              + f" -> {output_dim}")
        print(f"[FFNN] hidden activation: {hidden_activation}")
        print(f"[FFNN] output activation: {output_activation}")
        print(f"[FFNN] loss function: {loss_name}")
    
    # forward propagation
    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    # backward propagation
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        loss = self.loss_layer.forward(y_pred, y_true)
        grad = self.loss_layer.backward(y_pred, y_true)

        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        return loss
    
    # def update_params(self, lr: float = 0.01):
    #     "update w and b to all linear layer with the gradient stored"
    #     for layer in self.layers:
    #         if isinstance(layer, Linear):
    #             layer.w -= lr * layer.dw
    #             layer.b -= lr * layer.db
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, optimizer) -> float:
        y_batch = np.asarray(y_batch, dtype=np.float64).reshape(-1, 1)
        y_pred = self.forward(x_batch)
        loss = self.backward(y_pred, y_batch)
        optimizer.step(self.layers)
        return loss

    # predictttttt
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def predict(self, x: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int).ravel()

def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]

if __name__ == "__main__":
    from utils.data_loader import DataLoader
 
    # 1. Load & preprocess data
    loader = (
        DataLoader("data/datasetml_2026.csv")
        .load()
        .eda()
        .split(train_ratio=0.8, random_seed=42)
        .preprocess()
    )
 
    X_train, y_train = loader.get_train()
    X_test,  y_test  = loader.get_test()

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
 
    print(f"X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"X_test  : {X_test.shape}   |  y_test  : {y_test.shape}\n")
 
    # 2. Inisialisasi model
    #    Ganti hidden_activation: 'relu', 'tanh', 'leaky_relu', 'softplus'
    #    Ganti loss_name: 'binary_cross_entropy', 'mse'
    model = FFNN(
        input_dim         = X_train.shape[1],
        hidden_dim        = [128, 64],
        output_dim        = 1,
        hidden_activation = 'relu',
        output_activation = 'sigmoid',
        loss_name         = 'bce',
        # act_kwargs      = {'alpha': 0.05}   # aktifkan jika pakai leaky_relu
    )
 
    # 3. Training loop
    EPOCHS     = 30
    BATCH_SIZE = 32
    LR         = 0.01

    optimizer = GradientDescent(lr=LR)
 
    for epoch in range(1, EPOCHS + 1):
        batch_losses = []
 
        for X_batch, y_batch in batch_generator(X_train, y_train, batch_size=BATCH_SIZE):
            loss = model.train_step(X_batch, y_batch, optimizer)
            batch_losses.append(loss)
 
        if epoch % 5 == 0 or epoch == 1:
            avg_loss = np.mean(batch_losses)
            acc      = np.mean(model.predict(X_test) == y_test)
            print(f"Epoch {epoch:3d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Test Acc: {acc:.4f}")
 
    # 4. Evaluasi akhir
    final_acc = np.mean(model.predict(X_test) == y_test)
    print(f"\n[FFNN] Final Test Accuracy: {final_acc:.4f}")
