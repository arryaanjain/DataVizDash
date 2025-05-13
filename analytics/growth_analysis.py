"""
Growth analysis components for the application.
This module provides functionality to calculate year-over-year growth and delta metrics.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_utils import detect_date_columns, create_download_link
from datetime import datetime

def calculate_growth_metrics(data_dict):
    """
    Calculate growth metrics (delta and growth percentage) between consecutive years.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with years as keys and values as the metric to analyze
        Example: {"2021": 10.5, "2022": 10.3, "2024": 9, "2025": 15}

    Returns:
    --------
    growth_metrics : dict
        Dictionary with year pairs as keys and growth metrics as values
    """
    # Convert to sorted list of (year, value) tuples
    data_items = sorted([(int(year), value) for year, value in data_dict.items()])

    # Find the maximum value in the historical data (excluding the last year if it's a forecast)
    historical_max = max([value for _, value in data_items[:-1]]) if len(data_items) > 1 else data_items[0][1]

    # Calculate growth metrics for consecutive years
    growth_metrics = {}

    for i in range(len(data_items) - 1):
        current_year, current_value = data_items[i]
        next_year, next_value = data_items[i + 1]

        # Handle missing years (gap years)
        year_diff = next_year - current_year
        year_label = f"{current_year}-{next_year}"

        # Calculate delta and growth percentage
        delta = next_value - current_value

        # Avoid division by zero
        if current_value != 0:
            growth_percentage = (delta / current_value) * 100
        else:
            growth_percentage = float('inf') if delta > 0 else float('-inf') if delta < 0 else 0

        # Determine status based on delta
        if delta > 0:
            status = "Increasing"
        elif delta < 0:
            status = "Decreasing"
        else:
            status = "Stable"

        # Check for unusual growth or decline
        is_unusual = False
        remark = None

        # If growth percentage is extreme (more than 50% in either direction)
        if abs(growth_percentage) > 50:
            is_unusual = True
            if growth_percentage > 0:
                status = "Unusual increase"
                # Check if the next value exceeds historical maximum
                if next_value > historical_max and i == len(data_items) - 2:  # If it's the last pair and potentially a forecast
                    remark = f"Unusual growth. Historical max is {historical_max}. Claimed value is {next_value}."
            else:
                status = "Unusual decrease"
                remark = f"Unusual decline of {abs(growth_percentage):.2f}%."

        # Check for gap years
        if year_diff > 1:
            # Calculate annualized growth rate for gap years
            annualized_growth = ((next_value / current_value) ** (1 / year_diff) - 1) * 100

            # Add information about gap years
            if remark:
                remark += f" Note: {year_diff} year gap with annualized growth of {annualized_growth:.2f}%."
            else:
                remark = f"Note: {year_diff} year gap with annualized growth of {annualized_growth:.2f}%."

        # Store metrics
        growth_metrics[year_label] = {
            "delta": delta,
            "growth_percentage": growth_percentage,
            "status": status,
            "remark": remark
        }

    return growth_metrics

def predict_future_value(data_dict, target_year):
    """
    Predict a future value based on historical growth trends.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with years as keys and values as the metric to analyze
    target_year : int
        The year to predict

    Returns:
    --------
    predicted_value : float
        The predicted value for the target year
    method_used : str
        Description of the prediction method used
    """
    # Convert to sorted list of (year, value) tuples
    data_items = sorted([(int(year), value) for year, value in data_dict.items()])

    # Check if target year is already in the data
    if str(target_year) in data_dict:
        return data_dict[str(target_year)], "Existing value"

    # Check if we have enough data points for prediction
    if len(data_items) < 2:
        # Not enough data, use the only available value
        if len(data_items) == 1:
            return data_items[0][1], "Single data point (no growth applied)"
        return None, "Insufficient data"

    # Calculate average yearly growth (delta)
    total_delta = 0
    total_years = 0

    for i in range(len(data_items) - 1):
        current_year, current_value = data_items[i]
        next_year, next_value = data_items[i + 1]
        year_diff = next_year - current_year

        # Calculate delta and add to total
        delta = next_value - current_value
        total_delta += delta
        total_years += year_diff

    # Calculate average yearly delta
    avg_yearly_delta = total_delta / total_years if total_years > 0 else 0

    # Get the most recent data point
    last_year, last_value = data_items[-1]

    # Calculate years between last data point and target year
    years_to_predict = target_year - last_year

    # Predict using linear growth (average delta)
    predicted_value = last_value + (avg_yearly_delta * years_to_predict)

    return predicted_value, f"Linear growth based on average delta of {avg_yearly_delta:.2f} per year"

def create_growth_chart(data_dict, growth_metrics):
    """
    Create a visualization of the growth data.

    Parameters:
    -----------
    data_dict : dict
        Dictionary with years as keys and values as the metric to analyze
    growth_metrics : dict
        Dictionary with year pairs as keys and growth metrics as values

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
    # Convert to sorted list of (year, value) tuples
    data_items = sorted([(int(year), value) for year, value in data_dict.items()])
    years = [str(year) for year, _ in data_items]
    values = [value for _, value in data_items]

    # Create a figure with two subplots (value and growth percentage)
    fig = go.Figure()

    # Add the value line
    fig.add_trace(go.Scatter(
        x=years,
        y=values,
        mode='lines+markers',
        name='Value',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))

    # Add growth percentage bars on secondary y-axis
    growth_years = list(growth_metrics.keys())
    growth_percentages = [metrics['growth_percentage'] for metrics in growth_metrics.values()]

    # Create color list based on growth status
    colors = []
    for year_pair in growth_years:
        status = growth_metrics[year_pair]['status']
        if 'Unusual increase' in status:
            colors.append('darkred')  # Dark red for unusual increase
        elif 'Unusual decrease' in status:
            colors.append('darkblue')  # Dark blue for unusual decrease
        elif 'Increasing' in status:
            colors.append('green')  # Green for normal increase
        elif 'Decreasing' in status:
            colors.append('red')  # Red for normal decrease
        else:
            colors.append('gray')  # Gray for stable

    fig.add_trace(go.Bar(
        x=growth_years,
        y=growth_percentages,
        name='Growth %',
        marker_color=colors,
        yaxis='y2',
        opacity=0.7,
        text=[f"{pct:.1f}%" for pct in growth_percentages],
        textposition='outside'
    ))

    # Set up the layout with two y-axes
    fig.update_layout(
        title='Value and Growth Percentage by Year',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Value', side='left'),
        yaxis2=dict(
            title='Growth Percentage (%)',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified',
        barmode='group',
        height=500
    )

    return fig

def show_growth_analysis(df, numeric_cols):
    """
    Show growth analysis for selected numeric columns.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    numeric_cols : list
        List of numeric column names
    """
    st.subheader("Growth Trends Analysis")

    st.markdown("""
    This feature analyzes year-over-year growth and calculates delta metrics for your data.
    Select a numeric column and a date column to analyze growth trends over time.
    """)

    # Check if we have numeric columns
    if not numeric_cols:
        st.warning("No numeric columns found for growth analysis.")
        return

    # Detect date columns
    date_cols = detect_date_columns(df)

    if not date_cols:
        st.warning("No date columns found. Growth analysis requires a date column.")
        return

    # Create controls for column selection
    col1, col2 = st.columns(2)

    with col1:
        # Select metric column
        selected_metric = st.selectbox(
            "Select metric to analyze",
            numeric_cols,
            index=0
        )

    with col2:
        # Select date column
        selected_date = st.selectbox(
            "Select date column",
            date_cols,
            index=0
        )

    # Ensure date column is datetime
    df[selected_date] = pd.to_datetime(df[selected_date], errors='coerce')

    # Extract year from date column
    df['year'] = df[selected_date].dt.year

    # Group by year and calculate sum for the selected metric
    yearly_data = df.groupby('year')[selected_metric].sum().reset_index()

    # Convert to dictionary format for growth calculation
    data_dict = {str(int(row['year'])): row[selected_metric] for _, row in yearly_data.iterrows()}

    # Check if we have enough data
    if len(data_dict) < 2:
        st.warning(f"Not enough yearly data points for growth analysis. Found only {len(data_dict)} year(s).")

        # Show the available data
        st.write("Available data:")
        st.write(yearly_data)
        return

    # Calculate growth metrics
    growth_metrics = calculate_growth_metrics(data_dict)

    # Display the results
    st.write("### Growth Metrics")

    # Create a DataFrame for display
    growth_df = pd.DataFrame.from_dict(growth_metrics, orient='index')

    # Format the growth percentage
    growth_df['growth_percentage'] = growth_df['growth_percentage'].apply(lambda x: f"{x:.2f}%")

    # Display the growth metrics table
    st.dataframe(
        growth_df,
        use_container_width=True,
        column_config={
            "delta": st.column_config.NumberColumn(
                "Delta",
                format="%.2f"
            ),
            "growth_percentage": st.column_config.TextColumn(
                "Growth %"
            ),
            "status": st.column_config.TextColumn(
                "Status"
            ),
            "remark": st.column_config.TextColumn(
                "Remarks"
            )
        }
    )

    # Create and display the growth chart
    growth_chart = create_growth_chart(data_dict, growth_metrics)
    # Add a unique key to avoid duplicate element ID errors
    st.plotly_chart(growth_chart, use_container_width=True,
                   key=f"growth_chart_original_{selected_metric}")

    # Prediction section
    st.write("### Future Value Prediction")

    # Get current year
    current_year = datetime.now().year

    # Allow user to select a future year to predict
    future_year = st.slider(
        "Select year to predict",
        min_value=current_year,
        max_value=current_year + 10,
        value=current_year + 1
    )

    # Predict future value
    predicted_value, method = predict_future_value(data_dict, future_year)

    # Display prediction
    if predicted_value is not None:
        st.metric(
            label=f"Predicted {selected_metric} for {future_year}",
            value=f"{predicted_value:.2f}"
        )
        st.caption(f"Prediction method: {method}")

        # Add the prediction to the data dictionary for visualization
        data_dict_with_prediction = data_dict.copy()
        data_dict_with_prediction[str(future_year)] = predicted_value

        # Recalculate growth metrics with the prediction
        growth_metrics_with_prediction = calculate_growth_metrics(data_dict_with_prediction)

        # Create and display the updated chart
        updated_chart = create_growth_chart(data_dict_with_prediction, growth_metrics_with_prediction)
        # Add a unique key based on the prediction method and future year to avoid duplicate element ID errors
        st.plotly_chart(updated_chart, use_container_width=True,
                       key=f"growth_prediction_{method}_{future_year}_{selected_metric}")

        # Add a note about the prediction
        st.info("📊 The chart above includes the predicted value. The prediction is based on historical growth patterns.")
    else:
        st.error("Unable to make a prediction. Insufficient data.")

    # Add a download link for the growth data
    st.markdown("### Download Growth Analysis Data")

    # Create a DataFrame with the yearly data and growth metrics
    download_df = yearly_data.copy()

    # Add growth metrics to the download DataFrame
    for year_pair, metrics in growth_metrics.items():
        year_end = int(year_pair.split('-')[1])
        mask = download_df['year'] == year_end

        if any(mask):
            for key, value in metrics.items():
                if key != 'remark':  # Skip remarks for cleaner data
                    download_df.loc[mask, f"{key}"] = value

    # Add the download link
    st.markdown(create_download_link(download_df, "growth_analysis_data.csv",
                                   "Click here to download the growth analysis data"), unsafe_allow_html=True)
