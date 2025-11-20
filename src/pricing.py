# src/pricing.py
import numpy as np
import pandas as pd

def generate_candidates(base_price, min_price=150, max_price=600, step=25, max_discount_pct=0.3):
    low = max(min_price, base_price*(1-max_discount_pct))
    return np.arange(low, max_price+1, step)

def inventory_bias(stock, low_thresh=10, high_thresh=40, alpha=0.15, beta=0.1):
    if stock < low_thresh:
        return 1.0 + alpha * (low_thresh - stock)/low_thresh
    elif stock > high_thresh:
        return max(0.7, 1.0 - beta * (stock - high_thresh)/high_thresh)
    return 1.0

def choose_price(model, feature_cols, context_row, candidates):
    """Return dict with best price info. context_row: dict-like with keys required by features.prepare_features."""
    best = None
    best_score = -1e12
    for p in candidates:
        x = context_row.copy()
        x['price'] = float(p)
        x['price_ratio'] = p / max(1.0, x.get('competitor_price', 1.0))
        x['log_price'] = np.log(max(1.0, p))
        x['log_visitors'] = np.log1p(x.get('visitors', 0))
        # Make DataFrame with expected feature columns
        Xc = pd.DataFrame([x])[feature_cols].fillna(0)
        pred_sales = max(0.0, model.predict(Xc)[0])
        revenue = p * pred_sales
        bias = inventory_bias(int(x.get('stock', 0)))
        score = revenue * bias
        if score > best_score:
            best_score = score
            best = {'price': p, 'pred_sales': pred_sales, 'revenue': revenue, 'score': score}
    return best
