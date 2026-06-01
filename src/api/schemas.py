from pydantic import BaseModel


class ForecastRequest(BaseModel):
    """What the user sends to ask for a forecast."""
    day_of_week: int
    day_of_month: int
    month: int
    quarter: int
    week_of_year: int
    day_of_year: int
    is_weekend: int
    is_holiday: int
    days_since_start: int
    sales_lag_7: float
    sales_lag_14: float
    sales_lag_28: float
    sales_lag_365: float
    rolling_mean_7: float
    rolling_std_7: float
    rolling_mean_28: float
    rolling_std_28: float


class ForecastResponse(BaseModel):
    """What the API sends back."""
    predicted_sales: float