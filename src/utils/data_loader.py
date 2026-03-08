import os
import csv
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, file_path: str):
        # Inisialisasi DataLoader.
        self.file_path = file_path
        self.raw_data  = None   # Data mentah, diisi setelah load()
        self.X_train   = None
        self.X_test    = None
        self.y_train   = None
        self.y_test    = None
        self.X_val     = None
        self.y_val     = None

    def load(self) -> "DataLoader":

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"[DataLoader] Dataset tidak ditemukan di: '{self.file_path}'\n"
                f"Pastikan file CSV berada di dalam folder 'data/'"
            )

        self.raw_data = pd.read_csv(self.file_path)

        print(f"[DataLoader] Berhasil memuat {len(self.raw_data)} baris "
              f"dan {len(self.raw_data.columns)} kolom dari '{self.file_path}'")
        print(f"\n[DataLoader] Kolom: {self.raw_data.columns.tolist()}")

        return self

    def eda(self):
        
        # Gambaran Umum Dataset
        print("\n===Head Dataset===")
        print(self.raw_data.head(5))

        # Informasi Dataset
        print("\n===Datatypes Dataset===")
        print(self.raw_data.info())

        # Check missing data
        print("\nInformasi Missing Data")
        print(self.raw_data.isna().sum())

        # Check duplikasi data
        print("\nInformasi Duplikasi Data")
        print(self.raw_data.duplicated().sum())

        # Check Outlier


        # Informasi Distribusi Kelas Target :placement_status
        print("\n")
        print(self.raw_data['placement_status'].value_counts())
        print(self.raw_data['placement_status'].value_counts(normalize=True).round(3))

        return self

    # def preprocess(self) -> "DataLoader":

    
    # def handle_missing(self, data:r)


    # def split(self, train_ratio: float = 0.8, random_seed: int = 42) -> "DataLoader":

    # def get_data(self):


### Getter
    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_val(self):
        return self.X_val, self.y_val





