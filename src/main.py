from pathlib import Path
import sys
import os

SRC_ROOT = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from utils.data_loader import DataLoader
from model.layers import FFNN, Linear
from model.optimizer import GradientDescent
from utils.visualization import Visualizer


# Data
DATA_PATH   = '../data/datasetml_2026.csv'
TRAIN_RATIO = 0.8
VAL_SIZE    = 0.2
SEED        = 42

# Arsitektur
HIDDEN_DIM        = [128, 64]
HIDDEN_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'sigmoid'
LOSS_NAME         = 'bce'

# Weight Init
INIT_METHOD  = 'he'
DISTRIBUTION = None    # None → default
LOWER        = -0.5
UPPER        =  0.5
MEAN         =  0.0
VARIANCE     =  1.0

# Training
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 0.01

# Regularisasi (None = tidak pakai)
REG_TYPE   = None   # 'l1' | 'l2' | None
REG_LAMBDA = 0.01

# Output
OUTPUT_DIR  = '../results'
MODEL_NAME  = 'model'


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}.npz')

    print("=" * 65)
    print("FEEDFORWARD NEURAL NETWORK — IF3270 Pembelajaran Mesin")
    print("=" * 65)
    print("\n[1] PREPROCESSING DATA")
    print("-" * 65)

    loader = DataLoader(DATA_PATH)
    loader.load()
    loader.eda()
    loader.split(train_ratio=TRAIN_RATIO, random_seed=SEED)
    loader.preprocess()
    loader.split_val(val_size=VAL_SIZE, random_state=SEED)

    X_train, y_train = loader.get_train()
    X_val,   y_val   = loader.get_val()
    X_test,  y_test  = loader.get_test()

    X_train = np.asarray(X_train, dtype=np.float64)
    X_val   = np.asarray(X_val,   dtype=np.float64)
    X_test  = np.asarray(X_test,  dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val   = np.asarray(y_val,   dtype=np.float64)
    y_test  = np.asarray(y_test,  dtype=np.float64)

    input_dim = X_train.shape[1]

    print(f"\nInfo dataset:")
    print(f"  Fitur  : {input_dim}")
    print(f"  Train  : {X_train.shape[0]} sampel")
    print(f"  Val    : {X_val.shape[0]} sampel")
    print(f"  Test   : {X_test.shape[0]} sampel")
    print("\n[2] SETUP MODEL")
    print("-" * 65)

    model = FFNN(
        input_dim         = input_dim,
        hidden_dim        = HIDDEN_DIM,
        output_dim        = 1,
        hidden_activation = HIDDEN_ACTIVATION,
        output_activation = OUTPUT_ACTIVATION,
        loss_name         = LOSS_NAME,
        init_method       = INIT_METHOD,
        distribution      = DISTRIBUTION,
        seed              = SEED,
        lower             = LOWER,
        upper             = UPPER,
        mean              = MEAN,
        variance          = VARIANCE,
    )

    optimizer = GradientDescent(
        lr       = LR,
        reg_type = REG_TYPE,
        lam      = REG_LAMBDA,
    )

    print(f"\n[3] TRAINING ({EPOCHS} epochs)")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        metrics = model.train_epoch(
            X_train, y_train,
            optimizer  = optimizer,
            batch_size = BATCH_SIZE,
            x_val      = X_val,
            y_val      = y_val,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS}"
                f"  |  train_loss: {metrics['train_loss']:.4f}"
                f"  |  train_acc : {metrics['train_acc']:.4f}"
                f"  |  val_loss  : {metrics.get('val_loss', float('nan')):.4f}"
                f"  |  val_acc   : {metrics.get('val_acc',  float('nan')):.4f}"
            )

    print("\n[4] EVALUASI TEST SET")
    print("-" * 65)

    y_pred   = model.predict(X_test)
    test_acc = float(np.mean(y_pred == y_test.ravel()))

    tp = float(np.sum((y_pred == 1) & (y_test.ravel() == 1)))
    fp = float(np.sum((y_pred == 1) & (y_test.ravel() == 0)))
    fn = float(np.sum((y_pred == 0) & (y_test.ravel() == 1)))

    precision = tp / (tp + fp + 1e-15)
    recall    = tp / (tp + fn + 1e-15)
    f1        = 2 * precision * recall / (precision + recall + 1e-15)

    print(f"  Accuracy  : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")


    print("\n[5] SAVE MODEL")
    print("-" * 65)
    model.save(save_path)


    print("\n[6] VISUALISASI")
    print("-" * 65)

    viz = Visualizer()

    # Weight distribution
    try:
        weights = model.get_weight_distribution()
        viz.plot_weight_distribution(
            weight_dict = weights,
            title       = f"Weight Distribution — {MODEL_NAME}",
            save_path   = os.path.join(OUTPUT_DIR, 'weight_distribution.png'),
        )
        print("  Weight distribution tersimpan.")
    except Exception as e:
        print(f"  [WARNING] Tidak bisa plot weight: {e}")

    try:
        y_pred_viz = model.forward(X_test)
        model.backward(y_pred_viz, y_test.reshape(-1, 1))
        grads = model.get_gradient_distribution()
        viz.plot_gradient_distribution(
            grad_dict = grads,
            title     = f"Gradient Distribution — {MODEL_NAME}",
            save_path = os.path.join(OUTPUT_DIR, 'gradient_distribution.png'),
        )
        print("  Gradient distribution tersimpan.")
    except Exception as e:
        print(f"  [WARNING] Tidak bisa plot gradient: {e}")


    print("\n" + "=" * 65)
    print("RINGKASAN")
    print("=" * 65)
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Model         : {save_path}")
    print(f"  Output        : {OUTPUT_DIR}/")
    print(f" - weight_distribution.png")
    print(f" - gradient_distribution.png")
    print("=" * 65)

    return model, loader


if __name__ == "__main__":
    run()