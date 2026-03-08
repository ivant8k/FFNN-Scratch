import numpy as np
import pandas as pd
from utils.data_loader import DataLoader

def run():
    loader = DataLoader('data/datasetml_2026.csv')
    loader.load()

    # EDA (Explore Data Analyst)
    loader.eda()

    # loader.preprocess()

if __name__ == "__main__":
    run()