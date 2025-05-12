"""
Prophet forecasting components for the application.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from forecasting.forecast_utils import evaluate_forecast
from utils.data_utils import create_download_link

# Try to import Prophet, but handle the case where it's not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def show_prophet_forecast(ts_data, target_col):
    """Show Prophet forecast for time series data."""
    st.write("### Prophet Forecast")

    # Check if Prophet is available
    if not PROPHET_AVAILABLE:
        st.error("Prophet is not available. Please check your installation.")
        st.info("Prophet may require additional setup. See the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) for more information.")
        return

    # Forecast horizon
    prophet_periods = st.slider("Prophet forecast periods (months)", 3, 24, 12, key="prophet_periods")

    # Train Prophet model
    try:
        with st.spinner("Training Prophet model..."):
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': ts_data.index,
                'y': ts_data[target_col]
            })

            # Train-test split
            train_size = int(len(prophet_data) * 0.8)
            train_data = prophet_data.iloc[:train_size]
            test_data = prophet_data.iloc[train_size:]

            # Train model
            model = Prophet(yearly_seasonality=True,
                          weekly_seasonality=True,
                          daily_seasonality=False)
            model.fit(train_data)

            # Generate forecast for test period
            future = model.make_future_dataframe(periods=len(test_data), freq='M')
            forecast = model.predict(future)

            # Evaluate forecast
            test_forecast = forecast.tail(len(test_data))
            metrics = evaluate_forecast(test_data['y'].values, test_forecast['yhat'].values)

            # Display metrics
            st.write("#### Model Performance")
            metrics_df = pd.DataFrame({
                'Metric': list(metrics.keys()),
                'Value': list(metrics.values())
            })
            st.dataframe(metrics_df)

            # Train on full dataset for future forecast
            full_model = Prophet(yearly_seasonality=True,
                               weekly_seasonality=True,
                               daily_seasonality=False)
            full_model.fit(prophet_data)

            # Generate future forecast
            future = full_model.make_future_dataframe(periods=prophet_periods, freq='M')
            forecast = full_model.predict(future)

            # Plot components
            fig_comp = full_model.plot_components(forecast)
            st.write("#### Forecast Components")
            st.pyplot(fig_comp)

            # Plot forecast
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['y'],
                mode='lines',
                name='Historical Data',
                line=dict(color='blue')
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tail(prophet_periods),
                y=forecast['yhat'].tail(prophet_periods),
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))

            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'].tail(prophet_periods),
                           forecast['ds'].tail(prophet_periods).iloc[::-1]]),
                y=pd.concat([forecast['yhat_upper'].tail(prophet_periods),
                           forecast['yhat_lower'].tail(prophet_periods).iloc[::-1]]),
                fill='toself',
                fillcolor='rgba(0,176,246,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))

            fig.update_layout(
                title=f'Prophet Forecast for {target_col}',
                xaxis_title='Date',
                yaxis_title=target_col,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display forecast data
            st.write("#### Forecast Values")
            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prophet_periods)
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_display)

            # Download forecast
            if st.button("Download Prophet Forecast"):
                st.markdown(create_download_link(forecast_display,
                                               "prophet_forecast.csv",
                                               "Click here to download the forecast"),
                           unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in Prophet modeling: {e}")
        st.info("Prophet may require additional setup or a different data structure.")
