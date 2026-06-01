import pandas as pd
from config import config
from pathlib import Path

def load_raw_data()-> pd.DataFrame:
    """Load and store in the CSV"""
    df = pd.read_csv(config.TRAIN_CSV)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)#check if data/processed alreadt exists
    return df
def save_features(df:pd.DataFrame)->None:
    """Save the processed Data"""
    config.DATA_PROCESSED.mkdir(exist_ok=True)
    df.to_parquet(config.FEATURES_PARQUET, index=False)
def load_features() -> pd.DataFrame:
    return pd.read_parquet(config.FEATURES_PARQUET)

if __name__ == "__main__":
    df = load_raw_data()
    print("=" * 70)
    print(f"loaded data : {len(df)} rows")
    print(df.head())
    print(df.info())
    print("=" * 70)