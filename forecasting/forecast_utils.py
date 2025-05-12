"""
Utility functions for time series forecasting.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
# Import Prophet with error handling for NumPy 2.0 compatibility
try:
    # Try to patch NumPy for Prophet compatibility
    if hasattr(np, 'float_') is False and hasattr(np, 'float64'):
        np.float_ = np.float64
    from prophet import Prophet
except (ImportError, AttributeError) as e:
    import warnings
    warnings.warn(f"Prophet import error: {e}. Forecasting functionality may be limited.")
    Prophet = None
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_arima_model(data, p=1, d=1, q=0):
    """Train an ARIMA model on the data.

    Parameters:
    -----------
    data : pandas DataFrame or Series
        Time series data with datetime index
    p : int
        AR order (autoregressive)
    d : int
        Differencing order
    q : int
        MA order (moving average)

    Returns:
    --------
    model_fit : ARIMAResults
        Fitted ARIMA model
    """
    # Ensure data is properly formatted for ARIMA
    if isinstance(data, pd.DataFrame):
        # If it's a DataFrame, extract the first column as a Series
        if len(data.columns) > 0:
            # Get the column name (usually the target variable)
            col_name = data.columns[0]
            series = data[col_name]
        else:
            raise ValueError("DataFrame has no columns")
    else:
        # If it's already a Series, use it directly
        series = data

    # Create and fit the ARIMA model
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()

    # Store the original data index for later use in forecasting
    model_fit._index = series.index

    return model_fit

def train_prophet_model(data):
    """Train a Prophet model on the data."""
    # Create and train model
    model = Prophet(yearly_seasonality=True,
                   weekly_seasonality=True,
                   daily_seasonality=False)
    model.fit(data)

    return model

def generate_forecast(model, periods=12, model_type='arima'):
    """Generate forecast using the trained model."""
    if model_type == 'arima':
        forecast = model.forecast(steps=periods)

        # Get the last date from the model
        # In newer versions of statsmodels, the data structure has changed
        try:
            # First try our custom _index attribute
            if hasattr(model, '_index') and len(model._index) > 0:
                last_date = model._index[-1]
            # Then try the old way (model.data.index)
            elif hasattr(model, 'data') and hasattr(model.data, 'index'):
                last_date = model.data.index[-1]
            # Try to get it from model.model.data.row_labels (newer statsmodels)
            elif hasattr(model, 'model') and hasattr(model.model, 'data') and hasattr(model.model.data, 'row_labels'):
                last_date = model.model.data.row_labels[-1]
            # If all else fails, use current date
            else:
                import datetime
                last_date = datetime.datetime.now()
        except (AttributeError, IndexError):
            # Last resort: use a generic date
            import datetime
            last_date = datetime.datetime.now()

        # Create date range for forecast
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq='M'
        )

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast
        })
        forecast_df.set_index('date', inplace=True)

    elif model_type == 'prophet':
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        forecast_df = forecast_df.rename(columns={'ds': 'date', 'yhat': 'forecast',
                                                 'yhat_lower': 'lower_bound',
                                                 'yhat_upper': 'upper_bound'})
        forecast_df.set_index('date', inplace=True)

    return forecast_df

def evaluate_forecast(actual, predicted):
    """Evaluate forecast performance."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
