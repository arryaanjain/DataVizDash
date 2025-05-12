"""
Main forecasting components for the application.
"""
import streamlit as st
import pandas as pd
import traceback
import time
from utils.data_utils import detect_date_columns, prepare_time_series_data, validate_date_column
from config import FORECAST_TABS, MIN_FORECAST_DATA_POINTS

def show_forecasting(df, numeric_cols):
    """Show time series forecasting components."""
    st.subheader("Time Series Forecasting")

    try:
        # Detect date columns with improved accuracy
        date_cols = detect_date_columns(df)

        if not date_cols:
            st.warning("⚠️ No date columns detected in the data. Forecasting requires time series data.")
            st.info("Please upload a dataset with at least one date/time column.")

            # Show example of expected date format
            st.markdown("""
            ### Date Column Examples
            Your data should contain columns with dates in formats like:
            - `2023-01-15`
            - `01/15/2023`
            - `Jan 15, 2023`
            - `2023-01-15 14:30:00`
            """)
            return

        # Create a container for error messages
        error_container = st.empty()

        # Date column selection with validation
        date_col = st.selectbox(
            "Select date column",
            date_cols,
            help="Select a column that contains date/time values"
        )

        # Validate the selected date column
        is_valid_date, error_message = validate_date_column(df, date_col)

        if not is_valid_date:
            error_container.error(f"⚠️ {error_message}")
            st.info("Please select a different column that contains valid date values.")
            return

        # Clear any previous error messages
        error_container.empty()

        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        # Convert to datetime with proper error handling
        try:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')

            # Check for NaT values after conversion
            nat_count = df_copy[date_col].isna().sum()
            if nat_count > 0:
                st.warning(f"⚠️ {nat_count} rows ({nat_count/len(df_copy):.1%}) have invalid date values and will be excluded.")

            # Drop rows with NaT values
            df_copy = df_copy.dropna(subset=[date_col])

            if len(df_copy) == 0:
                st.error("❌ No valid date values found after conversion. Please select a different date column.")
                return

            # Calculate date range for information
            date_range = df_copy[date_col].max() - df_copy[date_col].min()
            date_min = df_copy[date_col].min().strftime('%Y-%m-%d')
            date_max = df_copy[date_col].max().strftime('%Y-%m-%d')

            st.success(f"✅ Date range: {date_min} to {date_max} ({date_range.days} days)")

            # Select target variable for forecasting
            if not numeric_cols:
                st.error("❌ No numeric columns found for forecasting.")
                return

            target_col = st.selectbox(
                "Select column to forecast",
                numeric_cols,
                help="Select a numeric column to forecast"
            )

            # Aggregation options
            aggregation_options = {
                'day': 'Daily',
                'week': 'Weekly',
                'month': 'Monthly',
                'quarter': 'Quarterly',
                'year': 'Yearly'
            }

            aggregation = st.selectbox(
                "Select time aggregation",
                options=list(aggregation_options.keys()),
                format_func=lambda x: aggregation_options[x],
                index=2,  # Default to monthly
                help="Aggregate data points to reduce noise and improve forecasting"
            )

            # Prepare time series data with proper error handling
            try:
                with st.spinner("Preparing time series data..."):
                    ts_data = prepare_time_series_data(df_copy, date_col, target_col, aggregation)

                # Display the time series data
                st.write(f"### Time Series Data ({aggregation_options[aggregation]} Aggregation)")

                # Show data points count
                st.info(f"📊 {len(ts_data)} data points available for forecasting")

                # Display chart
                st.line_chart(ts_data)

                # Check if we have enough data for forecasting
                if len(ts_data) >= MIN_FORECAST_DATA_POINTS:
                    # Create tabs for different forecasting methods
                    forecast_tabs = st.tabs(FORECAST_TABS)

                    # ARIMA Forecast tab
                    with forecast_tabs[0]:
                        try:
                            with st.spinner("Running ARIMA forecast..."):
                                # Import and run ARIMA forecast
                                from forecasting.arima import show_arima_forecast
                                show_arima_forecast(ts_data, target_col)
                        except Exception as e:
                            st.error(f"❌ Error in ARIMA forecasting: {str(e)}")
                            with st.expander("Error details", expanded=False):
                                st.code(traceback.format_exc())

                    # Prophet Forecast tab
                    with forecast_tabs[1]:
                        try:
                            with st.spinner("Running Prophet forecast..."):
                                # Import and run Prophet forecast
                                from forecasting.prophet import show_prophet_forecast
                                show_prophet_forecast(ts_data, target_col)
                        except Exception as e:
                            st.error(f"❌ Error in Prophet forecasting: {str(e)}")
                            with st.expander("Error details", expanded=False):
                                st.code(traceback.format_exc())


                else:
                    st.warning(f"⚠️ Not enough data for reliable forecasting. Found {len(ts_data)} data points, but at least {MIN_FORECAST_DATA_POINTS} are recommended.")
                    st.info("💡 Try using a different date column, changing the aggregation level, or uploading a dataset with more data points.")
            except Exception as e:
                st.error(f"❌ Error preparing time series data: {str(e)}")
                with st.expander("Error details", expanded=False):
                    st.code(traceback.format_exc())
        except Exception as e:
            st.error(f"❌ Error processing date column: {str(e)}")
            with st.expander("Error details", expanded=False):
                st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"❌ Unexpected error in forecasting module: {str(e)}")
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc())
