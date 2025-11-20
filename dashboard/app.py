# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# allow importing src package when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model import load_model
from src.pricing import generate_candidates, inventory_bias

# Paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "xgb_model.json")
FEAT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "feature_cols.npy")

# Load model and feature columns (with friendly errors)
if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please train model first (run src.model.train_and_save_model).")
    st.stop()

if not os.path.exists(FEAT_PATH):
    st.error(f"Feature columns file not found at {FEAT_PATH}. Please ensure training saved feature_cols.npy.")
    st.stop()

model = load_model(MODEL_PATH)
feature_cols = list(np.load(FEAT_PATH, allow_pickle=True))

st.title("Dynamic Pricing Dashboard (Prototype)")

st.sidebar.header("Context Inputs")
current_price = st.sidebar.number_input("Current price (₹):", value=400)
competitor_price = st.sidebar.number_input("Competitor price (₹):", value=420)
visitors = st.sidebar.number_input("Visitors:", value=220)
stock = st.sidebar.number_input("Stock level:", value=25)
promotion = st.sidebar.checkbox("Promotion active?")
day = st.sidebar.selectbox("Day of week", list(range(7)), index=0)
hour = st.sidebar.slider("Hour:", 0, 23, 12)
season = st.sidebar.selectbox("Season", ['normal','festive'])

# Build a single-row context dict matching training-time feature names where possible
context = {
    'price': None,  # filled per candidate
    'competitor_price': competitor_price,
    'price_ratio': None,  # filled per candidate
    'log_price': None,
    'log_visitors': np.log1p(visitors),
    'visitors': visitors,
    'stock': stock,
    'promotion': 1 if promotion else 0,
    'day_of_week': day,
    'hour': hour,
    'season_festive': 1 if season == 'festive' else 0,
    'stock_bucket_medium': 0,  # default, will be set below if needed
    'stock_bucket_high': 0
}

# Derive stock bucket flags same way as features.prepare_features
if stock > 20:
    context['stock_bucket_high'] = 1
    context['stock_bucket_medium'] = 0
elif stock > 5:
    context['stock_bucket_medium'] = 1
    context['stock_bucket_high'] = 0
else:
    context['stock_bucket_medium'] = 0
    context['stock_bucket_high'] = 0

# Generate candidates
candidates = generate_candidates(base_price=current_price)

results = []
for p in candidates:
    row = dict(context)  # shallow copy
    row['price'] = float(p)
    row['price_ratio'] = p / max(1.0, competitor_price)
    row['log_price'] = np.log(max(1.0, p))

    # Create DataFrame and ensure all training feature columns exist
    Xdf = pd.DataFrame([row])

    # Add any missing columns expected by the model, set to 0
    for col in feature_cols:
        if col not in Xdf.columns:
            Xdf[col] = 0

    # Reorder columns to match training order
    Xdf = Xdf[feature_cols].astype(float)

    # Predict
    try:
        pred_sales = float(model.predict(Xdf)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred_sales = max(0.0, pred_sales)
    revenue = p * pred_sales
    bias = inventory_bias(stock)
    score = revenue * bias
    results.append({'price': p, 'pred_sales': pred_sales, 'revenue': revenue, 'score': score})

res_df = pd.DataFrame(results).set_index('price')
best_price = int(res_df['score'].idxmax())

st.metric("Recommended Price (₹)", best_price)
st.write("Predicted sales:", round(res_df.loc[best_price, 'pred_sales'], 2))
st.line_chart(res_df[['pred_sales', 'revenue']])

st.subheader("Price vs Revenue (table)")
st.dataframe(res_df[['pred_sales', 'revenue', 'score']])

# Inventory-aware heatmap sample
st.subheader("Inventory-aware simulation (sample stocks)")
stocks_to_sim = [5, 15, 30, 60]
heat = []
for s in stocks_to_sim:
    row_vals = []
    for p in candidates:
        # recompute prediction for each (cheap enough for prototype)
        tmp = dict(context)
        tmp['price'] = float(p)
        tmp['price_ratio'] = p / max(1.0, competitor_price)
        tmp['log_price'] = np.log(max(1.0, p))
        tmp['stock'] = s
        # stock buckets
        sb_med = 1 if s > 5 and s <= 20 else 0
        sb_high = 1 if s > 20 else 0
        tmp['stock_bucket_medium'] = sb_med
        tmp['stock_bucket_high'] = sb_high

        Xtmp = pd.DataFrame([tmp])
        for col in feature_cols:
            if col not in Xtmp.columns:
                Xtmp[col] = 0
        Xtmp = Xtmp[feature_cols].astype(float)
        preds = float(model.predict(Xtmp)[0])
        revenue = p * max(0.0, preds)
        # apply inventory bias for this stock
        from src.pricing import inventory_bias as inv_bias
        score = revenue * inv_bias(s)
        row_vals.append(score)
    heat.append(row_vals)

heat_df = pd.DataFrame(heat, index=stocks_to_sim, columns=candidates)
st.write("Rows = stock levels, Columns = prices. Values = adjusted expected revenue")
st.dataframe(heat_df)
