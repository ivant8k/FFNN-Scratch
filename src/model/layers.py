from pathlib import Path
import sys
import os
import pickle
import json

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from model.activations import Activation
from model.loss import Loss
from model.initializer import Initializer

class Linear:
    def __init__(self, in_features: int, 
                 out_features: int,
                 init_method  : str  = 'zero',
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
            raise ValueError(f"[Linear] init_method '{init_method}' unrecognized. "
                f"Valid options: 'zero', 'uniform', 'normal', 'xavier', 'he'"
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
        deriv = deriv_fn(self.cache, **self.kwargs)
        if self.name == 'softmax':
            if np.asarray(deriv).ndim == 2:
                return dout @ deriv
            return np.einsum('bi,bij->bj', dout, deriv)
        return dout * deriv
    

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

        self._input_dim         = input_dim
        self._hidden_dim        = list(hidden_dim)
        self._output_dim        = output_dim
        self._hidden_activation = hidden_activation
        self._output_activation = output_activation
        self._loss_name         = loss_name
        self._init_method       = init_method

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
        print(f"[FFNN] weight init: {init_method}"
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

    def _prepare_targets_for_output(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
            return y.reshape(-1, 1)

        n_classes = y_pred.shape[1]

        # If labels are class indices, convert to one-hot.
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            labels = y.reshape(-1).astype(int)
            if np.any(labels < 0) or np.any(labels >= n_classes):
                raise ValueError(
                    f"[FFNN] multiclass target index out of range [0, {n_classes - 1}]"
                )
            return np.eye(n_classes, dtype=np.float64)[labels]

        if y.ndim == 2 and y.shape[1] == n_classes:
            return y

        raise ValueError(
            f"[FFNN] target shape {y.shape} incompatible with model output shape {y_pred.shape}"
        )

    def _targets_to_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] > 1:
            return np.argmax(y, axis=1)
        return y.ravel().astype(int)

    def _accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        y_pred_label = self.predict(x)
        y_true_label = self._targets_to_labels(y)
        return float(np.mean(y_pred_label == y_true_label))
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, optimizer) -> float:
        y_pred = self.forward(x_batch)
        y_batch = self._prepare_targets_for_output(y_batch, y_pred)
        loss = self.backward(y_pred, y_batch)
        optimizer.step(self.layers)
        return loss

    # predictttttt
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_proba = np.asarray(self.predict_proba(x), dtype=np.float64)

        if y_proba.ndim == 2 and y_proba.shape[1] > 1:
            return np.argmax(y_proba, axis=1)

        return (y_proba >= threshold).astype(int).ravel()

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
        train_acc  = self._accuracy(x_train, y_train)

        val_loss = val_acc = None
        if x_val is not None and y_val is not None:
            y_val_pred = self.forward(np.asarray(x_val, dtype=np.float64))
            y_val_true = self._prepare_targets_for_output(y_val, y_val_pred)
            val_loss   = self.loss_layer.forward(
                y_val_pred,
                y_val_true,
            )
            val_acc = self._accuracy(x_val, y_val)

        self._record_epoch(train_loss, train_acc, val_loss, val_acc)

        metrics = {'train_loss': train_loss, 'train_acc': train_acc}
        if val_loss is not None:
            metrics['val_loss'] = val_loss
            metrics['val_acc']  = val_acc
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Simpan model ke 2 file:
          - {filepath}.npz  → bobot semua Linear layer
          - {filepath}.json → konfigurasi arsitektur model
        """
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
 
        # 1. Simpan bobot ke .npz
        params  = {}
        lin_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                params[f'W_{lin_idx}'] = layer.w
                params[f'b_{lin_idx}'] = layer.b
                lin_idx += 1
        np.savez(filepath, **params)
 
        # 2. Simpan konfigurasi ke .json
        config = {
            'input_dim'         : self._input_dim,
            'hidden_dim'        : self._hidden_dim,
            'output_dim'        : self._output_dim,
            'hidden_activation' : self._hidden_activation,
            'output_activation' : self._output_activation,
            'loss_name'         : self._loss_name,
            'init_method'       : self._init_method,
        }
        with open(filepath + '.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
 
        print(f"[FFNN] Model tersimpan di '{filepath}.npz' dan '{filepath}.json'")
 
    @classmethod
    def load(cls, filepath: str) -> "FFNN":
        json_path = filepath + '.json'
        npz_path  = filepath + '.npz'
 
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[FFNN] Config tidak ditemukan: '{json_path}'")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"[FFNN] Bobot tidak ditemukan: '{npz_path}'")
 
        # 1. Load config dan rekonstruksi model
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
 
        model = cls(
            input_dim         = config['input_dim'],
            hidden_dim        = config['hidden_dim'],
            output_dim        = config['output_dim'],
            hidden_activation = config['hidden_activation'],
            output_activation = config['output_activation'],
            loss_name         = config['loss_name'],
            init_method       = config['init_method'],
        )
 
        # 2. Load bobot dari .npz
        data    = np.load(npz_path)
        lin_idx = 0
        for layer in model.layers:
            if isinstance(layer, Linear):
                key_w = f'W_{lin_idx}'
                key_b = f'b_{lin_idx}'
                if key_w not in data or key_b not in data:
                    raise KeyError(f"[FFNN] Key '{key_w}' atau '{key_b}' tidak ditemukan.")
                layer.w = data[key_w]
                layer.b = data[key_b]
                lin_idx += 1
 
        print(f"[FFNN] Model berhasil di-load dari '{filepath}' ")
        return model
    
    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            optimizer,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: int = 1,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
    ) -> dict:
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        for epoch in range(1, epochs + 1):
            metrics = self.train_epoch(
                X_train,
                y_train,
                optimizer,
                batch_size=batch_size,
                x_val=X_val,
                y_val=y_val,
            )

            if verbose == 1:
                bar_len = 30
                filled = int(bar_len * epoch / epochs)
                bar = '=' * filled + '-' * (bar_len - filled)
                t_loss = metrics['train_loss']
                t_acc = metrics['train_acc']
                val_part = ''
                if 'val_loss' in metrics:
                    val_part = f"val loss: {metrics['val_loss']:4f} - val_acc: {metrics['val_acc']:4f}"
                
                print(
                    f"\rEpoch {epoch:>{len(str(epochs))}}/{epochs} "
                    f"[{bar}]"
                    f" - loss: {t_loss:.4f} - acc: {t_acc:.4f}"
                    f"{val_part}",
                    end='\n' if epoch == epochs else '',
                    flush=True,
                )
        
        if verbose == 1:
            print() # newline

        return self.get_training_history()


def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]