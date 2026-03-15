import numpy as np

class Initializer:
    def zero(self, shape) -> np.ndarray:
        return np.zeros(shape)
    
    def uniform(self, shape, lower=-0.5, upper=0.5, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(lower, upper, shape)

    def normal(self, shape, mean=0.0, variance=1.0, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(mean, np.sqrt(variance), shape)
    
    def xavier(self, shape, distribution='uniform', seed=None) -> np.ndarray:
        fan_in, fan_out = shape
        rng = np.random.default_rng(seed)
        if distribution == 'uniform':
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, shape)
        elif distribution == 'normal':
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return rng.normal(0.0, std, shape)
        else:
            raise ValueError(f"[Initializer.xavier] distribution '{distribution}' tidak dikenal. Pilihan: 'uniform', 'normal'")

    def he(self, shape, distribution='normal', seed=None) -> np.ndarray:
        fan_in, _ = shape
        rng = np.random.default_rng(seed)
        if distribution == 'normal':
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, shape)
        elif distribution == 'uniform':
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, shape)
        else:
            raise ValueError(f"[Initializer.he] distribution '{distribution}' tidak dikenal. Pilihan: 'uniform', 'normal'")