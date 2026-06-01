import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from config import config

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Returns error as a percentage (lower = better).
    'Symmetric' = treats over- and under-prediction evenly,
    and avoids divide-by-zero blowups that plain MAPE has.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0          # if both are 0, error is 0 (not NaN)
    return 100.0 * np.mean(diff)

def seasonal_naive_baseline(df: pd.DataFrame) -> tuple:
    df = df.copy().sort_values(['store', 'item', 'date']).reset_index(drop = True)
    train_end = pd.to_datetime(config.TRAIN_END_DATE)
    valid_end = pd.to_datetime(config.VALID_TRAIN_END)

    # seasonal naive: predict each day from sales 7 days earlier. Compute on the
    # full series so the first 7 validation days use train data (not NaN).
    df['pred'] = df.groupby(['store', 'item'])['sales'].shift(7)
    valid_df = df[(df['date'] > train_end) & (df['date'] <= valid_end)].copy()
    
    y_true = valid_df['sales'].values
    y_pred = valid_df['pred'].values
    
    smape_val = smape(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    
    return y_true, y_pred, smape_val, mae_val

if __name__ == "__main__":
    from data import load_raw_data
    from features import engineer_features
    
    df = load_raw_data()
    df = engineer_features(df)
    y_true, y_pred, smape_val, mae_val = seasonal_naive_baseline(df)
    
    print(f"seasonal naive baseline")
    print(f"  SMAPE: {smape_val:.2f}%")
    print(f"  MAE:   {mae_val:.2f}")