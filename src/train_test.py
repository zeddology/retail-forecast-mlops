import pandas as pd
import numpy as np
import lightgbm as lgb
from config import config
from data import load_raw_data
from features import engineer_features
from evaluate import smape, seasonal_naive_baseline

df = load_raw_data()
df = engineer_features(df, horizon=config.FORECAST_HORIZON)

train_end = pd.to_datetime(config.TRAIN_END_DATE)
valid_end = pd.to_datetime(config.VALID_TRAIN_END)
train_df = df[df['date'] <= train_end].copy()
valid_df = df[(df['date'] > train_end) & (df['date'] <= valid_end)].copy()
print(f"Train: {len(train_df):,} | Valid: {len(valid_df):,}")

feature_cols = [c for c in df.columns if c not in ['date', 'sales', 'store', 'item']]
X_train, y_train = train_df[feature_cols], train_df['sales']
X_valid, y_valid = valid_df[feature_cols], valid_df['sales']

print("Training LightGBM... (this takes 10-60s)")
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                          num_leaves=63, random_state=42, verbose=-1)
model.fit(X_train, y_train)
print("Done training!")

y_pred = model.predict(X_valid)
print(f"Model SMAPE: {smape(y_valid.values, y_pred):.2f}%")
print(f"Baseline was: 17.34%")