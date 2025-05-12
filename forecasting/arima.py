"""
ARIMA forecasting components for the application.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from forecasting.forecast_utils import train_arima_model, generate_forecast, evaluate_forecast
from utils.data_utils import create_download_link

def show_arima_forecast(ts_data, target_col):
    """Show ARIMA forecast for time series data."""
    st.write("### ARIMA Forecast")

    # ARIMA parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        p = st.slider("p (AR order)", 0, 5, 1)
    with col2:
        d = st.slider("d (Differencing)", 0, 2, 1)
    with col3:
        q = st.slider("q (MA order)", 0, 5, 0)

    # Forecast horizon
    periods = st.slider("Forecast periods (months)", 3, 24, 12)

    # Train-test split for evaluation
    train_size = int(len(ts_data) * 0.8)
    train_data = ts_data.iloc[:train_size]
    test_data = ts_data.iloc[train_size:]

    # Train ARIMA model
    try:
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Train model on training data
        status_text.text("Training ARIMA model on training data...")
        progress_bar.progress(10)

        # Ensure train_data is properly formatted
        if isinstance(train_data, pd.DataFrame) and len(train_data.columns) > 0:
            # If it's a DataFrame with columns, we're good
            pass
        elif isinstance(train_data, pd.Series):
            # If it's a Series, convert to DataFrame
            train_data = pd.DataFrame(train_data)
        else:
            # If it's something else, raise an error
            raise ValueError("Training data must be a pandas DataFrame or Series")

        # Train the model
        model = train_arima_model(train_data, p, d, q)
        progress_bar.progress(30)

        # Step 2: Generate forecast for test period
        status_text.text("Generating forecast for test period...")
        test_forecast = model.forecast(steps=len(test_data))
        progress_bar.progress(50)

        # Step 3: Evaluate forecast
        status_text.text("Evaluating forecast performance...")

        # Ensure test_data is in the right format for evaluation
        if isinstance(test_data, pd.DataFrame):
            test_values = test_data.values
        elif isinstance(test_data, pd.Series):
            test_values = test_data.values
        else:
            test_values = test_data

        metrics = evaluate_forecast(test_values, test_forecast)
        progress_bar.progress(60)

        # Display metrics
        st.write("#### Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        st.dataframe(metrics_df)

        # Step 4: Train on full dataset for future forecast
        status_text.text("Training model on full dataset...")

        # Ensure ts_data is properly formatted
        if isinstance(ts_data, pd.DataFrame) and len(ts_data.columns) > 0:
            # If it's a DataFrame with columns, we're good
            pass
        elif isinstance(ts_data, pd.Series):
            # If it's a Series, convert to DataFrame
            ts_data = pd.DataFrame(ts_data)
        else:
            # If it's something else, raise an error
            raise ValueError("Time series data must be a pandas DataFrame or Series")

        full_model = train_arima_model(ts_data, p, d, q)
        progress_bar.progress(80)

        # Step 5: Generate future forecast
        status_text.text("Generating future forecast...")
        forecast_df = generate_forecast(full_model, periods, 'arima')
        progress_bar.progress(90)

        # Step 6: Create visualization
        status_text.text("Creating visualization...")

        # Plot historical data and forecast
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data[target_col],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue')
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f'ARIMA Forecast for {target_col}',
            xaxis_title='Date',
            yaxis_title=target_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Complete progress
        progress_bar.progress(100)
        status_text.text("Forecast complete!")

        st.plotly_chart(fig, use_container_width=True)

        # Display forecast data
        st.write("#### Forecast Values")
        st.dataframe(forecast_df)

        # Download forecast
        if st.button("Download ARIMA Forecast"):
            st.markdown(create_download_link(forecast_df.reset_index(),
                                           "arima_forecast.csv",
                                           "Click here to download the forecast"),
                       unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in ARIMA modeling: {e}")
        st.info("Try different parameters or a different model.")

        # Show detailed error information for debugging
        import traceback
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc())
