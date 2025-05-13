"""
Side-by-side chart components for the application.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import logging
from utils.data_utils import measure_time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log that this module is being imported
logger.info("Side-by-side charts module imported successfully")

@st.cache_data(ttl=3600)
@measure_time
def create_side_by_side_chart(df, x_col, y_cols, dimension_col, chart_type='bar', aggregation_method='mean'):
    """Create side-by-side charts for comparing dimensions.

    This function is used in the advanced_charts.py module to create side-by-side charts.
    If you're seeing this in the logs, the function is being called correctly.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    x_col : str
        The name of the x-axis column
    y_cols : list
        List of column names to plot on the y-axis
    dimension_col : str
        The column that defines the different dimensions to compare side by side
    chart_type : str, optional
        The type of chart to create ('bar' or 'line')
    aggregation_method : str, optional
        The method to use for aggregating values ('mean', 'sum', 'min', 'max')

    Returns:
    --------
    fig : plotly Figure
        The plotly figure object with side-by-side charts
    """
    # Log function call for debugging
    logger.info(f"Creating side-by-side chart with: dimension_col={dimension_col}, x_col={x_col}, y_cols={y_cols}, chart_type={chart_type}, aggregation_method={aggregation_method}")

    # Validate inputs
    if not isinstance(y_cols, list) or len(y_cols) == 0:
        y_cols = [y_cols]

    # Validate aggregation method
    valid_agg_methods = ['mean', 'sum', 'min', 'max']
    if aggregation_method not in valid_agg_methods:
        aggregation_method = 'mean'  # Default to mean if invalid

    # Get unique dimensions
    dimensions = df[dimension_col].unique()
    if len(dimensions) < 2:
        # Create a figure with a message if there's only one dimension
        fig = go.Figure()
        fig.add_annotation(
            text=f"Need at least 2 unique values in dimension column '{dimension_col}' for side-by-side comparison",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # Create a subplot figure with one subplot per dimension
    fig = go.Figure()

    # Calculate the number of rows and columns for the subplots
    if len(dimensions) <= 2:
        rows, cols = 1, len(dimensions)
    else:
        cols = min(2, len(dimensions))
        rows = (len(dimensions) + cols - 1) // cols  # Ceiling division

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"{dimension_col}: {dim}" for dim in dimensions],
        shared_yaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.2
    )

    # Create a color scale for better visual distinction
    colors = px.colors.qualitative.G10[:len(y_cols)]

    # Track min and max y values for consistent y-axis scaling
    y_min, y_max = float('inf'), float('-inf')

    # Process each dimension
    for i, dimension in enumerate(dimensions):
        # Calculate row and column for this subplot
        row = (i // cols) + 1
        col = (i % cols) + 1

        # Filter data for this dimension
        dim_df = df[df[dimension_col] == dimension].copy()

        # Skip if no data for this dimension
        if len(dim_df) == 0:
            continue

        # Sort data by x_col to ensure consistent ordering
        dim_df = dim_df.sort_values(by=x_col)

        # Get unique x values
        x_values = dim_df[x_col].unique()

        # Process each y column
        for j, y_col in enumerate(y_cols):
            # Skip if y_col not in dataframe
            if y_col not in dim_df.columns:
                continue

            # Use the selected aggregation method
            if aggregation_method == 'mean':
                y_values = [dim_df[dim_df[x_col] == x_val][y_col].mean() for x_val in x_values]
            elif aggregation_method == 'sum':
                y_values = [dim_df[dim_df[x_col] == x_val][y_col].sum() for x_val in x_values]
            elif aggregation_method == 'min':
                y_values = [dim_df[dim_df[x_col] == x_val][y_col].min() for x_val in x_values]
            elif aggregation_method == 'max':
                y_values = [dim_df[dim_df[x_col] == x_val][y_col].max() for x_val in x_values]

            # Update min and max for consistent y-axis
            if y_values:
                y_min = min(y_min, min([y for y in y_values if not pd.isna(y)]) if any(not pd.isna(y) for y in y_values) else y_min)
                y_max = max(y_max, max([y for y in y_values if not pd.isna(y)]) if any(not pd.isna(y) for y in y_values) else y_max)

            # Add the appropriate trace based on chart type
            if chart_type == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=x_values,
                        y=y_values,
                        name=y_col,
                        marker_color=colors[j],
                        text=[f"{y:.2f}" for y in y_values],
                        textposition='outside',
                        showlegend=i == 0,  # Only show in legend for first dimension
                    ),
                    row=row, col=col
                )
            elif chart_type == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        name=y_col,
                        mode='lines+markers',
                        marker=dict(color=colors[j]),
                        line=dict(color=colors[j]),
                        showlegend=i == 0,  # Only show in legend for first dimension
                    ),
                    row=row, col=col
                )

    # Set consistent y-axis range for all subplots if we have valid min/max
    if y_min != float('inf') and y_max != float('-inf'):
        y_padding = (y_max - y_min) * 0.1  # Add 10% padding
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if i * j <= len(dimensions):  # Only update valid subplots
                    fig.update_yaxes(range=[y_min - y_padding, y_max + y_padding], row=i, col=j)

    # Update layout
    agg_method_display = aggregation_method.upper()
    title = f'Side-by-Side Comparison by {dimension_col} ({agg_method_display})'

    fig.update_layout(
        title=title,
        barmode='group' if chart_type == 'bar' else None,
        height=max(500, 300 * rows),  # Adjust height based on number of rows
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2 / rows,  # Adjust based on number of rows
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=50, b=100, l=20, r=20),
    )

    # Update x-axis titles
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if i * j <= len(dimensions):  # Only update valid subplots
                fig.update_xaxes(title_text=x_col, row=i, col=j)
                if len(x_values) > 5:
                    fig.update_xaxes(tickangle=-45, row=i, col=j)

    # Update y-axis titles (only for leftmost subplots)
    for i in range(1, rows + 1):
        y_title = f"{', '.join(y_cols)} ({agg_method_display})"
        fig.update_yaxes(title_text=y_title, row=i, col=1)

    return fig
