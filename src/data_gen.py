# src/data_gen.py
import os
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_data.csv")

def generate_synthetic_data(n_days=180, periods_per_day=24, seed=42, out_path=OUT_PATH):
    """Generate data and write CSV only if file doesn't already exist."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"[data_gen] File already exists at {out_path} — skipping generation.")
        return out_path

    random.seed(seed); np.random.seed(seed)
    rows = []
    start = datetime(2024,1,1)
    product_id = 101
    for d in range(n_days):
        for h in range(periods_per_day):
            ts = start + timedelta(days=d, hours=h)
            day = ts.weekday()
            hour = ts.hour
            season = 'festive' if (d % 90) < 15 else 'normal'
            base_visitors = 200 + 30*np.sin(2*np.pi*(d%7)/7) + 20*np.sin(2*np.pi*hour/24)
            visitors = max(1, int(base_visitors + np.random.normal(0,15)))
            competitor_price = max(50, 400 + 50*np.sin(2*np.pi*d/30) + np.random.normal(0,20))
            stock = max(0, int(50 + 20*np.cos(2*np.pi*d/30) + np.random.normal(0,10)))
            promotion = 1 if random.random() < 0.05 else 0
            price = float(np.random.choice([200, 300, 350, 400, 450, 500]))
            # Hidden demand function
            price_effect = (price / 400.0) ** -1.6
            comp_effect = np.clip(competitor_price / price, 0.5, 2.0)
            inventory_effect = 1.0 if stock > 10 else 0.7 + 0.03*stock
            promotion_effect = 1.4 if promotion else 1.0
            season_effect = 1.6 if season == 'festive' else 1.0
            expected_sales = visitors * 0.05 * price_effect * (comp_effect**-0.5) * inventory_effect * promotion_effect * season_effect
            sales = max(0, int(np.random.poisson(max(0.2, expected_sales))))
            rows.append({
                'timestamp': ts, 'product_id': product_id, 'price': price,
                'competitor_price': round(competitor_price,2), 'visitors': visitors,
                'stock': stock, 'promotion': promotion, 'day_of_week': day,
                'hour': hour, 'season': season, 'sales': sales
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[data_gen] Synthetic data saved to {out_path} ({len(df)} rows)")
    return out_path

if __name__ == "__main__":
    generate_synthetic_data()
