from pathlib import Path
import sys
import os

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from model.activations import Activation
from model.loss import Loss
from model.optimizer import GradientDescent
from model.initializer import Initializer

class Linear:
    def __init__(self, in_features: int, 
                 out_features: int,
                 init_method  : str  = 'normal',
                 distribution : str  = None,
                 seed         : int  = None,
                 lower        : float = -0.5,
                 upper        : float =  0.5,
                 mean         : float =  0.0,
                 variance     : float =  1.0,):
        # w - weight matrix, b - bias vector
        init  = Initializer()
        shape = (in_features, out_features)

        if init_method == 'zero':
            self.w = init.zero(shape)
        elif init_method == 'uniform':
            self.w = init.uniform(shape, lower=lower, upper=upper, seed=seed)
        elif init_method == 'normal':
            self.w = init.normal(shape, mean=mean, variance=variance, seed=seed)
        elif init_method == 'xavier':
            dist   = distribution if distribution is not None else 'uniform'
            self.w = init.xavier(shape, distribution=dist, seed=seed)
        elif init_method == 'he':
            dist   = distribution if distribution is not None else 'normal'
            self.w = init.he(shape, distribution=dist, seed=seed)
        else:
            raise ValueError(
                f"[Linear] init_method '{init_method}' tidak dikenal. "
                f"Pilihan: 'zero', 'uniform', 'normal', 'xavier', 'he'"
            )
        
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
            # init
            init_method       : str  = 'normal',
            distribution      : str  = None,
            seed              : int  = None,
            lower             : float = -0.5,
            upper             : float =  0.5,
            mean              : float =  0.0,
            variance          : float =  1.0,
    ):
        self.layers = []
        act_kwargs = act_kwargs or {}

        # param untuk diteruskan ke linear
        init_kwargs = dict(
            init_method  = init_method,
            distribution = distribution,
            seed         = seed,
            lower        = lower,
            upper        = upper,
            mean         = mean,
            variance     = variance,
        )

        dims = [input_dim] + hidden_dim

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1], **init_kwargs))
            self.layers.append(ActivationLayer(hidden_activation, **act_kwargs))
        
        self.layers.append(Linear(dims[-1], output_dim, **init_kwargs))
        self.layers.append(ActivationLayer(output_activation))

        self.loss_layer = LossLayer(loss_name)
        self.history: dict = {
            'train_loss' : [],
            'val_loss'   : [],
            'train_acc'  : [],
            'val_acc'    : [],
        }

        print(f"[FFNN] architecture initialized with: {input_dim} -> "
              + f" -> ".join(str(d) for d in hidden_dim)
              + f" -> {output_dim}")
        print(f"[FFNN] hidden activation: {hidden_activation}")
        print(f"[FFNN] output activation: {output_activation}")
        print(f"[FFNN] loss function: {loss_name}")
        print(f"[FFNN] weight init  : {init_method}"
              + (f", seed={seed}" if seed is not None else ""))
    
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
    
    # Save and Load def save(self, filepath: str) -> None:
    def save(self, filepath: str) -> None:
        # Buat folder jika belum ada
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
 
        # Kumpulkan semua W dan b dari tiap Linear layer
        params = {}
        lin_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                params[f'W_{lin_idx}'] = layer.w
                params[f'b_{lin_idx}'] = layer.b
                lin_idx += 1
 
        # Simpan ke .npz
        np.savez(filepath, **params)
        print(f"[FFNN] Model tersimpan di '{filepath}' "
              f"({lin_idx} linear layer)")
 
    def load(self, filepath: str) -> "FFNN":
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[FFNN] File tidak ditemukan: '{filepath}'"
            )
 
        # Load file .npz
        data = np.load(filepath)
 
        # Restore W dan b ke tiap Linear layer
        lin_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                key_w = f'W_{lin_idx}'
                key_b = f'b_{lin_idx}'
 
                if key_w not in data or key_b not in data:
                    raise KeyError(
                        f"[FFNN] Key '{key_w}' atau '{key_b}' tidak ditemukan di file. "
                    )
 
                layer.w = data[key_w]
                layer.b = data[key_b]
                lin_idx += 1
 
        print(f"[FFNN] Model berhasil di-load dari '{filepath}' "
              f"({lin_idx} linear layer)")
        return self

    # visualization helpers
    def get_weight_distribution(self) -> dict:
        result  = {}
        lin_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                result[f'Linear_{lin_idx}'] = layer.w.ravel().copy()
                lin_idx += 1
        return result
    
    def get_gradient_distribution(self) -> dict:
        result  = {}
        lin_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                if layer.dw is not None:
                    result[f'Linear_{lin_idx}'] = layer.dw.ravel().copy()
                lin_idx += 1
        return result

    def get_training_history(self) -> dict:
        return {k: list(v) for k, v in self.history.items()}

    def get_validation_loss(self) -> list:
        return list(self.history['val_loss'])

    def _record_epoch(
        self,
        train_loss : float,
        train_acc  : float,
        val_loss   : float | None = None,
        val_acc    : float | None = None,
    ) -> None:
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)

    def train_epoch(
        self,
        x_train    : np.ndarray,
        y_train    : np.ndarray,
        optimizer,
        batch_size : int   = 32,
        x_val      : np.ndarray | None = None,
        y_val      : np.ndarray | None = None,
    ) -> dict:
        batch_losses = []
        for x_b, y_b in batch_generator(x_train, y_train, batch_size):
            loss = self.train_step(x_b, y_b, optimizer)
            batch_losses.append(loss)

        train_loss = float(np.mean(batch_losses))
        train_acc  = float(np.mean(self.predict(x_train) == y_train.ravel()))

        val_loss = val_acc = None
        if x_val is not None and y_val is not None:
            y_val_pred = self.forward(np.asarray(x_val, dtype=np.float64))
            val_loss   = self.loss_layer.forward(
                y_val_pred,
                np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
            )
            val_acc = float(np.mean(self.predict(x_val) == np.asarray(y_val).ravel()))

        self._record_epoch(train_loss, train_acc, val_loss, val_acc)

        metrics = {'train_loss': train_loss, 'train_acc': train_acc}
        if val_loss is not None:
            metrics['val_loss'] = val_loss
            metrics['val_acc']  = val_acc
        return metrics

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
        metrics = model.train_epoch(
            X_train, y_train,
            optimizer  = optimizer,
            batch_size = BATCH_SIZE,
            x_val      = X_test,   # pakai test set sebagai val set
            y_val      = y_test,
        )
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}"
                  f"  |  Train Loss: {metrics['train_loss']:.4f}"
                  f"  |  Val Loss: {metrics.get('val_loss', float('nan')):.4f}"
                  f"  |  Val Acc: {metrics.get('val_acc', float('nan')):.4f}")
 
    # 4. Ambil data untuk visualisasi
    weights  = model.get_weight_distribution()      # -> plot histogram bobot
    grads    = model.get_gradient_distribution()    # -> plot histogram gradien
    history  = model.get_training_history()         # -> plot train/val loss & acc
    val_loss = model.get_validation_loss()          # -> plot validation loss saja

    print(f"\n[FFNN] Layer weights  : {list(weights.keys())}")
    print(f"[FFNN] Layer grads    : {list(grads.keys())}")
    print(f"[FFNN] History keys   : {list(history.keys())}")
    print(f"[FFNN] Val loss len   : {len(val_loss)} epochs")
    print(f"\n[FFNN] Final Test Accuracy: {history['val_acc'][-1]:.4f}")
