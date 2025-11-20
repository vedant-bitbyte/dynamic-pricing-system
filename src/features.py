# src/features.py
import numpy as np
import pandas as pd

def prepare_features(df):
    """Return a new DataFrame with model features. Does NOT modify input df in-place."""
    df2 = df.copy()
    df2['price_ratio'] = df2['price'] / (df2['competitor_price'].replace(0, 1))
    df2['log_price'] = np.log(df2['price'].clip(lower=1))
    df2['log_visitors'] = np.log1p(df2['visitors'])
    df2['stock_bucket'] = pd.cut(df2['stock'], bins=[-1,5,20,1000], labels=['low','medium','high'])
    # one-hot season and stock_bucket
    df2 = pd.get_dummies(df2, columns=['season','stock_bucket'], drop_first=True)
    # ensure columns exist even if some dummies missing
    for col in ['season_festive','stock_bucket_medium','stock_bucket_high']:
        if col not in df2.columns:
            df2[col] = 0
    # keep relevant columns
    feature_cols = ['price','competitor_price','price_ratio','log_price','log_visitors',
                    'visitors','stock','promotion','day_of_week','hour',
                    'season_festive','stock_bucket_medium','stock_bucket_high']
    return df2, feature_cols
