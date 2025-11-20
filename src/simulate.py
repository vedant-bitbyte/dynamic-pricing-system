# src/simulate.py
import os
import pandas as pd
import numpy as np
from .features import prepare_features
from .model import load_model
from .pricing import generate_candidates, choose_price
from .data_gen import generate_synthetic_data

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "xgb_model.json")

def simulate_test(csv_path='data/synthetic_data.csv', model_path=MODEL_PATH, out_csv=None):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if out_csv is None:
        out_csv = os.path.join(ARTIFACT_DIR, "decisions.csv")
    df = pd.read_csv(csv_path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    # time split: last 30% as test
    split_idx = int(len(df) * 0.7)
    df_test = df.iloc[split_idx:].copy()
    model = load_model(model_path)
    _, feature_cols = prepare_features(df_test)  # we use columns set (function returns consistent list)
    decisions = []
    for _, row in df_test.iterrows():
        context = {
            'competitor_price': row['competitor_price'],
            'visitors': row['visitors'],
            'stock': int(row['stock']),
            'promotion': int(row['promotion']),
            'day_of_week': int(row['day_of_week']),
            'hour': int(row['hour']),
            'season_festive': 1 if row['season']=='festive' else 0
        }
        candidates = generate_candidates(base_price=row['price'])
        best = choose_price(model, feature_cols, context, candidates)
        # For synthetic data we can simulate actual sales with a simple proxy:
        # here we will use the historical hidden function shape: (approximate)
        # NOTE: in real data you cannot simulate actual sales; you would log actuals later in production
        true_sales = simulate_true_sales(row, best['price'])  # implemented below
        decisions.append({
            'timestamp': row['timestamp'],
            'historic_price': row['price'],
            'chosen_price': best['price'],
            'pred_sales': best['pred_sales'],
            'pred_revenue': best['revenue'],
            'actual_sales_sim': true_sales,
            'actual_revenue_sim': true_sales * best['price']
        })
    outdf = pd.DataFrame(decisions)
    outdf.to_csv(out_csv, index=False)
    print(f"[simulate] Saved decisions to {out_csv}")
    return outdf

def simulate_true_sales(row, price):
    """Recreate the hidden demand used in data_gen (approx) to simulate actual sales for chosen price."""
    visitors = row['visitors']
    competitor_price = row['competitor_price']
    stock = row['stock']
    promotion = row['promotion']
    season = row['season']
    # hidden model (same shape as data_gen)
    price_effect = (price / 400.0) ** -1.6
    comp_effect = np.clip(competitor_price / price, 0.5, 2.0)
    inventory_effect = 1.0 if stock > 10 else 0.7 + 0.03*stock
    promotion_effect = 1.4 if promotion else 1.0
    season_effect = 1.6 if season == 'festive' else 1.0
    expected_sales = visitors * 0.05 * price_effect * (comp_effect**-0.5) * inventory_effect * promotion_effect * season_effect
    return max(0, int(np.random.poisson(max(0.2, expected_sales))))

if __name__ == "__main__":
    # ensure data exists
    generate_synthetic_data()
    simulate_test()
