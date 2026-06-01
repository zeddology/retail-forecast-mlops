import lightgbm as lgb 
import mlflow
import mlflow.lightgbm
import pandas as pd
import numpy as np
from config import config
from data import load_raw_data, save_features
from features import engineer_features
from evaluate import smape, seasonal_naive_baseline

def train_model():
    print("loading...")
    df = load_raw_data()
    df = engineer_features(df, horizon=config.FORECAST_HORIZON)
    save_features(df)

    train_end = pd.to_datetime(config.TRAIN_END_DATE)
    valid_end = pd.to_datetime(config.VALID_TRAIN_END)
    train_df = df[df['date'] <= train_end].copy()
    valid_df = df[(df['date'] > train_end) & (df['date'] <= valid_end)].copy()
    print(f"train : {len(train_df):,} rows | valid: {len(valid_df):,} rows")

    #drop non feature columns
    #this decides what goes into training and what doesnt
    feature_cols = [c for c in df.columns if c not in ['date', 'sales', 'store', 'item']]
    X_train, y_train = train_df[feature_cols], train_df['sales']
    X_valid, y_valid = valid_df[feature_cols], valid_df['sales']

    _, _, smape_base, mae_base = seasonal_naive_baseline(df)

    #mlflow tracking
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
    }

    with mlflow.start_run():
        # Train
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
        )

        # Evaluate
        y_pred = model.predict(X_valid)
        smape_val = smape(y_valid.values, y_pred)
        mae_val = np.mean(np.abs(y_valid.values - y_pred))

        print(f"\nModel SMAPE: {smape_val:.2f}%  (baseline {smape_base:.2f}%)")
        print(f"Model MAE:   {mae_val:.2f}")
        print(f"Beat baseline? {'YES ✓' if smape_val < smape_base else 'NO ✗'}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_param("horizon", config.FORECAST_HORIZON)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metric("baseline_smape", smape_base)
        mlflow.log_metric("smape", smape_val)
        mlflow.log_metric("mae", mae_val)
        mlflow.lightgbm.log_model(model, "model")

        print(f"\nLogged to MLflow run: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train_model()
