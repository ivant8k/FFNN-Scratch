import numpy as np
import pandas as pd
from utils.data_loader import DataLoader

def run():
    loader = DataLoader('data/datasetml_2026.csv')
    loader.load()

    # EDA (Explore Data Analyst)
    loader.eda()

    # Split data train test dan preprocess
    loader.split().preprocess()
    loader.split_val()

    X_train, y_train = loader.get_train()
    X_test, y_test = loader.get_test()
    X_val, y_val = loader.get_val()

    print("\n=== Verifikasi Dimensi Matriks ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")


if __name__ == "__main__":
    run()