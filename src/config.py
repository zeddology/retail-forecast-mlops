from pathlib import Path
from pydantic_settings import BaseSettings

class Config(BaseSettings):
    #paths 
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    
    #DATA
    TRAIN_CSV: Path = DATA_RAW / "train.csv"
    FEATURES_PARQUET : Path = DATA_PROCESSED / "features.parquet"
    #TRAIN
    TRAIN_END_DATE: str = "2016-12-31"#train up to this date
    VALID_TRAIN_END: str = "2017-12-31"#validate up to this date
    FORECAST_HORIZON: int = 7 #Forecast next 7 days
    
    #ML_FLOW
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT: str = "retail-demand"
    
    #model
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
config = Config()
    
    