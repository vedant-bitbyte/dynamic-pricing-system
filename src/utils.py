# src/utils.py
import pandas as pd
import os

def read_data(path="data/synthetic_data.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, parse_dates=['timestamp'])
