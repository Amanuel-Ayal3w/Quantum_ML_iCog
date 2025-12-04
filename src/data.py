import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_data(test_size: float = 0.2, random_state: int = 42,
              url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
              save_path: str = "data/heart_disease.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], MinMaxScaler]:
   

    columns = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach',
               'exang','oldpeak','slope','ca','thal','target']

    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        # Check if the file has a header by reading the first line
        with open(save_path, 'r') as f:
            first_line = f.readline().strip()
        
        if first_line.startswith('age,sex'):
             df = pd.read_csv(save_path, na_values='?')
        else:
             df = pd.read_csv(save_path, names=columns, na_values='?')
    else:
        print(f"Downloading dataset from {url}...")
        df = pd.read_csv(url, header=None, names=columns, na_values='?')
        # Save a copy locally with headers
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path} with headers")

    # Drop rows with missing values for simplicity
    df = df.dropna().reset_index(drop=True)

    # Map target: 0 -> 0 (no disease), >0 -> 1 (disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Consider numeric columns only (exclude target)
    feature_cols = [c for c in df.columns if c != 'target']

    X_all = df[feature_cols].astype(float)
    y_all = df['target'].astype(int)


    selected = ['age', 'chol', 'thal', 'cp']

    X_sel = X_all[selected].values

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_scaled = scaler.fit_transform(X_sel)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all.values, test_size=test_size, random_state=random_state, stratify=y_all.values
    )

    return X_train, X_test, y_train, y_test, selected, scaler


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, selected, scaler = load_data()
    print(f"Selected features: {selected}")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


