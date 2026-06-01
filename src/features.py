import pandas as pd
import numpy as np
from datetime import datetime
import holidays                  # library that knows public holidays by country
from config import config        # our paths + settings


def engineer_features(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Turn the 4 raw columns (date, store, item, sales) into a rich feature set
    the model can actually learn from.

    'horizon' = how many days ahead we forecast (7 = predict one week out).
    We shift lag/rolling features by 'horizon' so the model never peeks at
    data it wouldn't have when making a real forecast (leakage prevention).
    """
    df = df.copy().sort_values(['store', 'item', 'date']).reset_index(drop=True)
    df['day_of_week']  = df['date'].dt.dayofweek      # 0=Mon ... 6=Sun (weekly cycle)
    df['day_of_month'] = df['date'].dt.day            # 1..31 (pay-day / month-end effects)
    df['month']        = df['date'].dt.month          # 1..12 (seasonality)
    df['quarter']      = df['date'].dt.quarter        # 1..4 (broad seasonal blocks)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)  # 1..52
    df['day_of_year']  = df['date'].dt.dayofyear      # 1..365 (fine-grained seasonality)

    # Weekend flag — sales behave differently Sat/Sun. 1 if weekend, else 0.
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x in us_holidays else 0)
    
    df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
    
    #add lag features to see the sales N days ago
    
    for lag in [7, 14, 28, 365]:
        df[f'sales_lag_{lag}'] = (df.groupby(['store', 'item'])['sales'].shift(lag+horizon))
        
    #Introduce nrolling features:
    # window 7  = last week's level / volatility
    # window 28 = last month's level / volatility
    #.transform ensures that after the calculation the size of df remains the same
    for window in [7, 28]:
        df[f'rolling_mean_{window}'] = (
            df.groupby(['store', 'item'])['sales']
            .transform(lambda x : x.shift(horizon).rolling(window, min_periods=1).mean())
        )
        df[f'rolling_std_{window}'] = (df.groupby(['store', 'item'])['sales']
                    .transform(lambda x : x.shift(horizon).rolling(window, min_periods= 1).std())
                    )
    df = df.fillna(0)
    return df


if __name__ == "__main__":
    from data import load_raw_data, save_features

    df = load_raw_data()                                   # 1. load clean raw data
    df = engineer_features(df, horizon=config.FORECAST_HORIZON)  # 2. add features
    save_features(df)                                      # 3. save to parquet

    print(f"Engineered {len(df)} rows with {len(df.columns)} columns")
    print(df.columns.tolist())                             # show all feature names
    print(df.head())  