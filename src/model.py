# src/model.py
import os
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import prepare_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")

class XGBBoosterWrapper:
    """
    Wraps an xgboost.Booster to provide a .predict(X_df) method that accepts a pandas DataFrame.
    """
    def __init__(self, booster, feature_cols):
        self.booster = booster
        self.feature_cols = list(feature_cols)

    def predict(self, X_df):
        # ensure DataFrame has required columns in right order
        Xc = X_df.copy()
        # If any feature is missing, add it as zeros
        for c in self.feature_cols:
            if c not in Xc.columns:
                Xc[c] = 0
        Xc = Xc[self.feature_cols]
        dmat = xgb.DMatrix(Xc.values, feature_names=self.feature_cols)
        preds = self.booster.predict(dmat)
        # return numpy array like sklearn predict
        return preds

def train_and_save_model(csv_path, save_path=MODEL_PATH, num_rounds=1000, early_stopping_rounds=20):
    """
    Train using xgb.train (Booster) to allow early_stopping_rounds reliably across xgboost versions.
    Returns the feature column list used for training.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df_feat, feature_cols = prepare_features(df)
    X = df_feat[feature_cols]
    y = df_feat['sales'].astype(float)

    split_idx = int(len(df_feat) * 0.7)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train.values, label=y_train.values, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val.values, label=y_val.values, feature_names=feature_cols)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05,
        'max_depth': 6,
        'verbosity': 0
    }

    evals = [(dtrain, 'train'), (dval, 'valid')]
    booster = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=evals,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

    # Evaluate on validation
    preds_val = booster.predict(dval)
    mae = mean_absolute_error(y_val, preds_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds_val))  # compatible with all sklearn versions
    print(f"[model] Val MAE: {mae:.4f}")
    print(f"[model] Val RMSE: {rmse:.4f}")

    # Save the booster
    booster.save_model(save_path)
    print(f"[model] Saved booster to {save_path}")

    # Save feature column order (useful to keep inference stable)
    feat_path = os.path.join(MODEL_DIR, "feature_cols.npy")
    np.save(feat_path, np.array(feature_cols))
    print(f"[model] Saved feature column order to {feat_path}")

    return feature_cols

def load_model(path=MODEL_PATH):
    """
    Loads the saved booster and its feature columns and returns a wrapper with .predict(X_df)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    booster = xgb.Booster()
    booster.load_model(path)
    feat_path = os.path.join(os.path.dirname(path), "feature_cols.npy")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"Feature column file not found: {feat_path}")
    feature_cols = list(np.load(feat_path, allow_pickle=True))
    return XGBBoosterWrapper(booster, feature_cols)
