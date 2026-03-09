import os
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def __init__(self, file_path: str):
        # Inisialisasi DataLoader.
        self.file_path = file_path
        self.raw_data  = None   # Data mentah, diisi setelah load()
        self.train_df  = None
        self.test_df   = None
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
        print("\n===Informasi Outlier (Metode IQR)===")
        # Ambil semua kolom dengan tipe data numerik
        num_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        
        for col in num_cols:
            # Panggil fungsi helper menggunakan self dan oper data dari self.raw_data
            lower, upper = self.get_iqr_bounds(self.raw_data[col])
            n = ((self.raw_data[col] < lower) | (self.raw_data[col] > upper)).sum()
            print(f"{col}: {n} outliers")

        # Informasi Distribusi Kelas Target :placement_status
        print("\n Informasi Distribusi Target")
        print(self.raw_data['placement_status'].value_counts())
        print(self.raw_data['placement_status'].value_counts(normalize=True).round(3))

        return self

    def preprocess(self) -> "DataLoader":
        if self.train_df is None:
            raise ValueError("Jalankan split() sebelum preprocess() untuk mencegah data leakage.")
        # Handling Missing data (tidak perlu tidak ada data yang hilang)

        # Handling Duplicate data (Tidak perlu tidak ada data yang duplikat)

        # Handling Outlier (Clipping)
        NUM_COLS = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in NUM_COLS:
            lower, upper = self.get_iqr_bounds(self.train_df[col])
            self.train_df[col] = self.train_df[col].clip(lower, upper)
            self.test_df[col] = self.test_df[col].clip(lower, upper)

        print("\n")
        print("====Check outlier after clipping====")
        for col in NUM_COLS:
            lower, upper = self.get_iqr_bounds(self.train_df[col])
            n = ((self.train_df[col] < lower) | (self.train_df[col] > upper)).sum()
            print(f"{col}: {n} outliers")

        # Encoding Fitur Kategorikal
        OHE_CAT_COLS = ['country', 'specialization', 'industry']
       
        # 1. One-Hot Encoding
        self.train_df = pd.get_dummies(self.train_df, columns=OHE_CAT_COLS, drop_first=True)
        self.test_df = pd.get_dummies(self.test_df, columns=OHE_CAT_COLS, drop_first=True)

        # Align data
        train_cols = self.train_df.columns
        self.test_df = self.test_df.reindex(columns=train_cols, fill_value=0)

        # 2. Ordinal Encoding
        tier_map = {'Tier 3': 1, 'Tier 2': 2, 'Tier 1': 3}
        rank_map = {'300+': 1, '100-300': 2, 'Top 100': 3}
        target_map = {'Not Placed': 0, 'Placed': 1}

        for df in [self.train_df, self.test_df]:
            df['college_tier'] = df['college_tier'].map(tier_map)
            df['university_ranking_band'] = df['university_ranking_band'].map(rank_map)
            df['placement_status'] = df['placement_status'].map(target_map)

        # Scaling
        NUM_COLS_TO_SCALE = [c for c in NUM_COLS if c in self.train_df.columns and c != 'placement_status']
        scaler = StandardScaler()
        
        self.train_df[NUM_COLS_TO_SCALE] = scaler.fit_transform(self.train_df[NUM_COLS_TO_SCALE])
        self.test_df[NUM_COLS_TO_SCALE]  = scaler.transform(self.test_df[NUM_COLS_TO_SCALE])

        # Pisah target
        self.y_train = self.train_df['placement_status'].values
        self.y_test = self.test_df['placement_status'].values
        
        self.X_train = self.train_df.drop(columns=['placement_status']).values
        self.X_test = self.test_df.drop(columns=['placement_status']).values

        print("[DataLoader] Preprocessing selesai.")

        return self

    def split(self, train_ratio: float = 0.8, random_seed: int = 42) -> "DataLoader":
        if self.raw_data is None:
            raise ValueError("[DataLoader] Data mentah belum dimuat. Jalankan load() terlebih dahulu.")
            
        # Mengacak indeks
        np.random.seed(random_seed)
        shuffled_indices = np.random.permutation(len(self.raw_data))
        train_size = int(len(self.raw_data) * train_ratio)
        
        train_indices = shuffled_indices[:train_size]
        test_indices = shuffled_indices[train_size:]
        
        self.train_df = self.raw_data.iloc[train_indices].copy()
        self.test_df = self.raw_data.iloc[test_indices].copy()
        
        print(f"[DataLoader] Data di-split: {len(self.train_df)} Train, {len(self.test_df)} Test")

        return self

    # def visualisasi():

### Getter
    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_val(self):
        return self.X_val, self.y_val

    # Helper
    def get_iqr_bounds(self, series):
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR






