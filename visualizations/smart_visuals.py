"""
Smart visualization components for the application.
This module provides intelligent chart selection and automatic column detection.
Enhanced with value display features for all chart types.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.data_utils import detect_date_columns
from config import SMART_CHART_TYPES

def detect_best_columns_for_chart(df, chart_type, numeric_cols, categorical_cols):
    """
    Automatically detect the most appropriate columns for a given chart type.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    chart_type : str
        The type of chart to create
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names

    Returns:
    --------
    dict
        Dictionary with recommended columns for the chart
    """
    recommendations = {}

    # Detect date columns
    date_cols = detect_date_columns(df)

    # Find columns with low cardinality (good for grouping)
    low_cardinality_cols = []
    for col in categorical_cols:
        if len(df[col].unique()) <= 10:  # Columns with 10 or fewer unique values
            low_cardinality_cols.append(col)

    # Find columns with high cardinality (good for IDs or details)
    high_cardinality_cols = []
    for col in categorical_cols:
        if len(df[col].unique()) > 10:
            high_cardinality_cols.append(col)

    # Find numeric columns with high variance (good for analysis)
    high_variance_cols = []
    for col in numeric_cols:
        if df[col].std() > 0:
            high_variance_cols.append((col, df[col].std()))

    # Sort by variance (highest first)
    high_variance_cols.sort(key=lambda x: x[1], reverse=True)
    high_variance_cols = [col for col, _ in high_variance_cols]

    # Chart-specific recommendations
    if chart_type == "Pie Chart":
        # For pie charts, we need one categorical column with low-to-medium cardinality
        # and one numeric column for values
        recommendations["category_col"] = low_cardinality_cols[0] if low_cardinality_cols else categorical_cols[0] if categorical_cols else None
        recommendations["value_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0] if numeric_cols else None

    elif chart_type == "Bar Chart":
        # For bar charts, we need one categorical column and one numeric column
        recommendations["x_col"] = low_cardinality_cols[0] if low_cardinality_cols else categorical_cols[0] if categorical_cols else None
        recommendations["y_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0] if numeric_cols else None
        # Optional color grouping
        if len(low_cardinality_cols) > 1:
            recommendations["color_col"] = low_cardinality_cols[1]

    elif chart_type == "Line Chart":
        # For line charts, we need one x-axis column (preferably numeric or date) and one y-axis column
        if date_cols:
            recommendations["x_col"] = date_cols[0]
        elif numeric_cols:
            # Find a numeric column that could be an x-axis (e.g., years, ids)
            for col in numeric_cols:
                if df[col].nunique() > 5 and df[col].is_monotonic_increasing:
                    recommendations["x_col"] = col
                    break
            if "x_col" not in recommendations:
                recommendations["x_col"] = numeric_cols[0]
        else:
            recommendations["x_col"] = categorical_cols[0] if categorical_cols else None

        # Y-axis should be numeric and different from x-axis
        y_options = [col for col in high_variance_cols if col != recommendations.get("x_col")]
        recommendations["y_col"] = y_options[0] if y_options else numeric_cols[0] if numeric_cols else None

        # Optional color grouping
        if low_cardinality_cols:
            recommendations["color_col"] = low_cardinality_cols[0]

    elif chart_type == "Time Series":
        # For time series, we need a date column and a numeric column
        recommendations["x_col"] = date_cols[0] if date_cols else None
        recommendations["y_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0] if numeric_cols else None
        # Optional color grouping
        if low_cardinality_cols:
            recommendations["color_col"] = low_cardinality_cols[0]

    elif chart_type == "Scatter Plot":
        # For scatter plots, we need two numeric columns
        if len(numeric_cols) >= 2:
            recommendations["x_col"] = numeric_cols[0]
            recommendations["y_col"] = numeric_cols[1]
            # Optional color grouping
            if low_cardinality_cols:
                recommendations["color_col"] = low_cardinality_cols[0]
            # Optional size column
            if len(numeric_cols) >= 3:
                recommendations["size_col"] = numeric_cols[2]

    elif chart_type == "Heatmap":
        # For heatmaps, we need two categorical columns and one numeric column
        if len(categorical_cols) >= 2 and numeric_cols:
            recommendations["x_col"] = categorical_cols[0]
            recommendations["y_col"] = categorical_cols[1]
            recommendations["value_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0]

    elif chart_type == "Box Plot":
        # For box plots, we need one categorical column and one numeric column
        recommendations["x_col"] = low_cardinality_cols[0] if low_cardinality_cols else categorical_cols[0] if categorical_cols else None
        recommendations["y_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0] if numeric_cols else None

    elif chart_type == "Stacked Bar Chart":
        # For stacked bar charts, we need two categorical columns and one numeric column
        if len(categorical_cols) >= 2:
            recommendations["x_col"] = categorical_cols[0]
            recommendations["color_col"] = categorical_cols[1]
            recommendations["y_col"] = high_variance_cols[0] if high_variance_cols else numeric_cols[0] if numeric_cols else None

    return recommendations

def create_smart_chart(df, chart_type, chart_options):
    """
    Create a chart based on the selected type and options.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    chart_type : str
        The type of chart to create
    chart_options : dict
        Dictionary with options for the chart

    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The created chart
    """
    if chart_type == "Pie Chart":
        category_col = chart_options.get("category_col")
        value_col = chart_options.get("value_col")

        if not category_col or not value_col:
            return None

        # Aggregate data
        value_counts = df.groupby(category_col)[value_col].sum().reset_index()

        # If there are too many categories, keep only top 5 and group others
        if len(value_counts) > 5:
            top_5 = value_counts.nlargest(5, value_col)
            others_sum = value_counts[~value_counts[category_col].isin(top_5[category_col])][value_col].sum()
            others_row = pd.DataFrame({category_col: ["Others"], value_col: [others_sum]})
            value_counts = pd.concat([top_5, others_row], ignore_index=True)

        fig = px.pie(
            value_counts,
            names=category_col,
            values=value_col,
            title=f'Distribution of {value_col} by {category_col}',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3,  # Make it a donut chart
            custom_data=[value_col]  # Add custom data for hover
        )

        # Improve layout with enhanced value display
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<br>Percentage: %{percent}<extra></extra>'
        )

        # Add total value in the center of donut
        total_value = value_counts[value_col].sum()

        # Use go.Figure to access the figure for advanced customization
        fig_go = go.Figure(fig)
        fig_go.add_annotation(
            text=f"Total<br>{total_value:.2f}",
            x=0.5, y=0.5,
            font=dict(size=14, color="black", family="Arial"),
            showarrow=False
        )
        fig = fig_go

        fig.update_layout(
            margin=dict(t=50, b=80, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )

        # Add summary statistics below the chart
        fig.add_annotation(
            text=f"Total: {total_value:.2f} | Categories: {len(value_counts)}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        return fig

    elif chart_type == "Bar Chart":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")

        if not x_col or not y_col:
            return None

        # Aggregate data - no color grouping
        agg_df = df.groupby(x_col)[y_col].sum().reset_index()

        # Calculate statistics for summary
        total_value = agg_df[y_col].sum()
        avg_value = agg_df[y_col].mean()
        max_value = agg_df[y_col].max()
        min_value = agg_df[y_col].min()

        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            title=f'{y_col} by {x_col}',
            text=y_col,  # Add value labels
            custom_data=[agg_df[y_col]]  # Add custom data for enhanced hover
        )

        # Format the text labels with improved visibility
        fig.update_traces(
            texttemplate='<b>%{y:.2f}</b>',
            textposition='outside',
            textfont=dict(size=11, color="black", family="Arial"),
            hovertemplate='<b>%{x}</b><br>' +
                          f'{y_col}: ' + '%{y:.2f}<br>' +
                          f'Percentage of Total: %{{customdata[0] / {total_value} * 100:.1f}}%' +
                          '<extra></extra>'
        )

        # Improve layout with more information
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=50, b=100, l=20, r=20),
        )

        # Add summary statistics below the chart
        fig.add_annotation(
            text=f"Total: {total_value:.2f} | Average: {avg_value:.2f} | Max: {max_value:.2f} | Min: {min_value:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        # Ensure x-axis labels are properly aligned
        if len(df[x_col].unique()) > 5:
            fig.update_xaxes(tickangle=-45)

        return fig

    elif chart_type == "Time Series":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")
        color_col = chart_options.get("color_col")
        user_aggregation = chart_options.get("aggregation", "auto")

        if not x_col or not y_col:
            return None

        # Sort data by x-axis
        df_sorted = df.sort_values(by=x_col)

        # For time series, ensure x-axis is datetime
        is_datetime = False
        if chart_type == "Time Series":
            try:
                df_sorted[x_col] = pd.to_datetime(df_sorted[x_col])
                is_datetime = True
            except:
                pass

        # Check if we need to apply automatic data aggregation
        aggregation_applied = False
        aggregation_message = ""

        # For Time Series charts, apply intelligent data handling
        if is_datetime or chart_type == "Time Series":
            # Define threshold for automatic aggregation
            AGGREGATION_THRESHOLD = 5000

            # Check if dataset exceeds threshold
            if len(df_sorted) > AGGREGATION_THRESHOLD:
                from utils.data_utils import determine_optimal_aggregation

                # If user selected "auto", determine optimal aggregation
                if user_aggregation == "auto":
                    try:
                        optimal_aggregation, aggregation_message = determine_optimal_aggregation(df_sorted, x_col)

                        # Apply aggregation if needed
                        if optimal_aggregation != 'none':
                            # Group by time period and calculate mean
                            try:
                                # Create a copy of the dataframe with only numeric columns for aggregation
                                numeric_columns = df_sorted.select_dtypes(include=['number']).columns.tolist()

                                # Make sure y_col is in the numeric columns
                                if y_col not in numeric_columns:
                                    st.warning(f"Column '{y_col}' is not numeric and cannot be aggregated.")
                                    raise ValueError(f"Column '{y_col}' is not numeric")

                                # Set the index and resample
                                if optimal_aggregation == 'day':
                                    df_sorted = df_sorted.set_index(x_col).resample('D')[numeric_columns].mean().reset_index()
                                elif optimal_aggregation == 'week':
                                    df_sorted = df_sorted.set_index(x_col).resample('W')[numeric_columns].mean().reset_index()
                                elif optimal_aggregation == 'month':
                                    df_sorted = df_sorted.set_index(x_col).resample('M')[numeric_columns].mean().reset_index()
                                elif optimal_aggregation == 'quarter':
                                    df_sorted = df_sorted.set_index(x_col).resample('Q')[numeric_columns].mean().reset_index()
                                elif optimal_aggregation == 'year':
                                    df_sorted = df_sorted.set_index(x_col).resample('Y')[numeric_columns].mean().reset_index()

                                aggregation_applied = True
                            except Exception as e:
                                # If resampling fails, log the error and continue without aggregation
                                st.warning(f"Automatic aggregation failed: {str(e)}. Using raw data instead.")
                                aggregation_message = "Automatic aggregation failed. Using raw data."
                    except Exception as e:
                        # If optimal aggregation detection fails, continue without aggregation
                        st.warning(f"Could not determine optimal aggregation: {str(e)}. Using raw data instead.")
                        aggregation_message = "Could not determine optimal aggregation. Using raw data."
                # If user selected a specific aggregation level
                elif user_aggregation != 'none':
                    # Apply user-selected aggregation
                    try:
                        # Create a copy of the dataframe with only numeric columns for aggregation
                        numeric_columns = df_sorted.select_dtypes(include=['number']).columns.tolist()

                        # Make sure y_col is in the numeric columns
                        if y_col not in numeric_columns:
                            st.warning(f"Column '{y_col}' is not numeric and cannot be aggregated.")
                            raise ValueError(f"Column '{y_col}' is not numeric")

                        # Set the index and resample
                        if user_aggregation == 'day':
                            df_sorted = df_sorted.set_index(x_col).resample('D')[numeric_columns].mean().reset_index()
                        elif user_aggregation == 'week':
                            df_sorted = df_sorted.set_index(x_col).resample('W')[numeric_columns].mean().reset_index()
                        elif user_aggregation == 'month':
                            df_sorted = df_sorted.set_index(x_col).resample('M')[numeric_columns].mean().reset_index()
                        elif user_aggregation == 'quarter':
                            df_sorted = df_sorted.set_index(x_col).resample('Q')[numeric_columns].mean().reset_index()
                        elif user_aggregation == 'year':
                            df_sorted = df_sorted.set_index(x_col).resample('Y')[numeric_columns].mean().reset_index()

                        aggregation_applied = True
                        aggregation_message = f"Data aggregated to {user_aggregation}ly level as selected"
                    except Exception as e:
                        # If resampling fails, log the error and continue without aggregation
                        st.warning(f"{user_aggregation.capitalize()} aggregation failed: {str(e)}. Using raw data instead.")
                        aggregation_message = f"{user_aggregation.capitalize()} aggregation failed. Using raw data."

        # Determine if we should add markers based on data size
        use_markers = len(df_sorted) < 100

        # Calculate statistics for summary
        # Check if the dataframe is not empty and y_col exists and is numeric
        if len(df_sorted) > 0 and y_col in df_sorted.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(df_sorted[y_col]):
                avg_value = df_sorted[y_col].mean()
                max_value = df_sorted[y_col].max()
                min_value = df_sorted[y_col].min()
                last_value = df_sorted[y_col].iloc[-1] if len(df_sorted) > 0 else None
            else:
                # For non-numeric columns, we can't calculate statistics
                avg_value = max_value = min_value = 0
                last_value = None
                st.warning(f"Column '{y_col}' is not numeric. Statistics cannot be calculated.")
        else:
            # If dataframe is empty or column doesn't exist
            avg_value = max_value = min_value = 0
            last_value = None

        # Create the chart with enhanced data display
        fig = px.line(
            df_sorted,
            x=x_col,
            y=y_col,
            title=f'Trend of {y_col} over {x_col}',
            markers=use_markers,
            custom_data=[df_sorted[y_col]]  # Add custom data for enhanced hover
        )

        # Add improved hover information
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' +
                          f'{y_col}: ' + '%{y:.2f}<extra></extra>'
        )

        # Add value labels for key points if dataset isn't too large
        # Only add labels if the column is numeric
        if len(df_sorted) > 0 and y_col in df_sorted.columns and pd.api.types.is_numeric_dtype(df_sorted[y_col]):
            if len(df_sorted) < 50:
                # Add all points
                fig.update_traces(
                    textposition="top center",
                    texttemplate="%{y:.2f}",
                    textfont=dict(size=10)
                )
            else:
                # Add labels only for key points (min, max, last)
                try:
                    max_idx = df_sorted[y_col].idxmax()
                    min_idx = df_sorted[y_col].idxmin()
                    last_idx = df_sorted.index[-1]

                    # Create unique indices list (in case min/max/last are the same point)
                    key_indices = list(set([max_idx, min_idx, last_idx]))

                    for idx in key_indices:
                        try:
                            point_x = df_sorted.loc[idx, x_col]
                            point_y = df_sorted.loc[idx, y_col]
                            label = "Max" if idx == max_idx else "Min" if idx == min_idx else "Latest"

                            # Only add annotation if point_y is a number
                            if pd.notnull(point_y) and isinstance(point_y, (int, float)):
                                fig.add_annotation(
                                    x=point_x,
                                    y=point_y,
                                    text=f"{label}: {point_y:.2f}",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowsize=1,
                                    arrowwidth=1,
                                    arrowcolor="#636363",
                                    font=dict(size=10, color="black"),
                                    bgcolor="white",
                                    bordercolor="#c7c7c7",
                                    borderwidth=1,
                                    borderpad=4
                                )
                        except Exception as e:
                            # If there's an error with a specific point, skip it
                            st.warning(f"Could not add label for a key point: {str(e)}")
                            continue
                except Exception as e:
                    # If there's an error finding key points, skip the labeling
                    st.warning(f"Could not identify key points for labeling: {str(e)}")
                    pass

        # Improve layout with more space at the top to avoid overlapping with range selector buttons
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=70, b=70, l=20, r=20),  # Increased margins to accommodate annotations
            hovermode="closest"
        )

        # Add summary statistics below the chart
        # Format the last_value properly, handling None case
        if last_value is not None:
            last_value_text = f"{last_value:.2f}"
        else:
            last_value_text = "N/A"

        fig.add_annotation(
            text=f"Average: {avg_value:.2f} | Max: {max_value:.2f} | Min: {min_value:.2f} | Latest: {last_value_text}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        # Add annotation about data aggregation if applied - positioned to avoid overlap with range selector
        if aggregation_applied and aggregation_message:
            try:
                # Add as a subtitle instead of an annotation to avoid overlap with range selector buttons
                # Get the current title text safely
                current_title = f'Trend of {y_col} over {x_col}'
                if hasattr(fig, 'layout') and hasattr(fig.layout, 'title'):
                    if hasattr(fig.layout.title, 'text') and fig.layout.title.text:
                        current_title = fig.layout.title.text

                # Create the subtitle with the aggregation message
                subtitle = f"<br><span style='font-size:10px;color:gray'>{aggregation_message}</span>"

                # Update the title with the subtitle
                fig.update_layout(
                    title={
                        'text': current_title + subtitle,
                        'y': 0.95,  # Position the title a bit higher
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'
                    }
                )
            except Exception as e:
                # If updating the title fails, add a separate annotation instead
                st.warning(f"Could not update chart title: {str(e)}")
                fig.add_annotation(
                    text=aggregation_message,
                    xref="paper", yref="paper",
                    x=0.5, y=1.05,
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    align="center"
                )

        # Ensure x-axis labels are properly aligned
        if chart_type != "Time Series" and len(df_sorted[x_col].unique()) > 5:
            fig.update_xaxes(tickangle=-45)

        # For Time Series charts, add range selector and slider for better navigation
        if chart_type == "Time Series" and is_datetime:
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    # Position the range selector to avoid overlap with the title
                    y=1.0,
                    x=0.0,
                    yanchor="top",
                    xanchor="left"
                )
            )

        # Instead of storing attributes directly on the figure object,
        # we'll use Plotly's layout metadata to store our custom information
        if aggregation_applied and aggregation_message:
            # Store the aggregation info in the figure's layout metadata
            # Use custom properties in layout that won't interfere with Plotly's functionality
            # Avoid using underscore prefixes as they're not allowed in Plotly layout
            fig.update_layout(
                meta_aggregation_info={
                    "applied": True,
                    "message": aggregation_message
                }
            )

        return fig

    elif chart_type == "Scatter Plot":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")
        color_col = chart_options.get("color_col")
        size_col = chart_options.get("size_col")

        if not x_col or not y_col:
            return None

        # Create a copy of the dataframe to avoid modifying the original
        plot_df = df.copy()

        # Ensure we have valid data for the scatter plot
        plot_df = plot_df.dropna(subset=[x_col, y_col])

        # Handle size column if specified
        if size_col:
            # Check if size column has valid values (must be positive)
            size_values = plot_df[size_col].copy()

            # If values are negative, try to transform them
            if (size_values < 0).any():
                # Option 1: Use absolute values
                size_values = size_values.abs()

                # Option 2: If all values are negative, invert them
                if (size_values <= 0).all():
                    size_values = size_values.max() - size_values + 1

            # Ensure minimum size
            size_values = size_values.clip(lower=5)

            # Create a new column for size
            plot_df['_size_col_'] = size_values
            size_col_to_use = '_size_col_'
        else:
            size_col_to_use = None

        try:
            # Calculate statistics for summary
            avg_x = plot_df[x_col].mean()
            avg_y = plot_df[y_col].mean()
            corr = plot_df[[x_col, y_col]].corr().iloc[0, 1]

            # No color grouping - removed as requested
            if size_col_to_use:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    size=size_col_to_use,
                    title=f'{y_col} vs {x_col} (size: {size_col})',
                    opacity=0.7,
                    trendline="ols",  # Add trendline
                    custom_data=[plot_df.index]  # Add index for hover
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f'{y_col} vs {x_col}',
                    opacity=0.7,
                    trendline="ols",  # Add trendline
                    custom_data=[plot_df.index]  # Add index for hover
                )

            # Enhance hover information
            for i in range(len(fig.data)):
                if "trendline" not in fig.data[i].name:  # Skip trendline traces
                    fig.data[i].hovertemplate = (
                        f'<b>{x_col}</b>: %{{x:.2f}}<br>' +
                        f'<b>{y_col}</b>: %{{y:.2f}}<br>' +
                        (f'<b>{size_col}</b>: %{{marker.size}}<br>' if size_col_to_use else '') +
                        '<extra></extra>'
                    )

            # Add labels for outlier points
            if len(plot_df) < 100:  # Only for smaller datasets
                # Calculate z-scores to find outliers
                from scipy import stats

                try:
                    # Calculate z-scores for both x and y
                    z_scores_x = np.abs(stats.zscore(plot_df[x_col]))
                    z_scores_y = np.abs(stats.zscore(plot_df[y_col]))

                    # Find points that are outliers in either dimension
                    outlier_indices = np.where((z_scores_x > 2.5) | (z_scores_y > 2.5))[0]

                    # Add annotations for outliers
                    for idx in outlier_indices:
                        point_x = plot_df.iloc[idx][x_col]
                        point_y = plot_df.iloc[idx][y_col]

                        fig.add_annotation(
                            x=point_x,
                            y=point_y,
                            text=f"({point_x:.1f}, {point_y:.1f})",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="#636363",
                            font=dict(size=9, color="black"),
                            bgcolor="white",
                            bordercolor="#c7c7c7",
                            borderwidth=1,
                            borderpad=2
                        )
                except:
                    # If z-score calculation fails, skip outlier detection
                    pass

            # Add trendline equation and correlation
            if len(plot_df) >= 2:  # Need at least 2 points for correlation
                # Format correlation coefficient
                corr_text = f"Correlation: {corr:.2f}"

                # Add correlation annotation
                fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    text=corr_text,
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.3)",
                    borderwidth=1,
                    borderpad=4
                )

            # Improve layout
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                margin=dict(t=50, b=70, l=20, r=20),
                hovermode="closest"
            )

            # Add summary statistics below the chart
            fig.add_annotation(
                text=f"Avg {x_col}: {avg_x:.2f} | Avg {y_col}: {avg_y:.2f} | Correlation: {corr:.2f}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color="gray")
            )

            return fig
        except Exception as e:
            # If scatter plot fails, return a simple fallback chart
            st.warning(f"Could not create scatter plot: {str(e)}")

            # Create a simple fallback chart
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                title=f'{y_col} vs {x_col} (simplified)',
                opacity=0.7
            )

            return fig

    elif chart_type == "Heatmap":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")
        value_col = chart_options.get("value_col")

        if not x_col or not y_col or not value_col:
            return None

        # Aggregate data
        pivot_df = df.pivot_table(
            index=y_col,
            columns=x_col,
            values=value_col,
            aggfunc='mean'
        ).fillna(0)

        # Calculate statistics for summary
        avg_value = pivot_df.mean().mean()
        max_value = pivot_df.max().max()
        min_value = pivot_df.min().min()

        # Create heatmap with enhanced value display
        fig = px.imshow(
            pivot_df,
            labels=dict(x=x_col, y=y_col, color=value_col),
            x=pivot_df.columns,
            y=pivot_df.index,
            title=f'Heatmap of {value_col} by {x_col} and {y_col}',
            color_continuous_scale='Viridis',
            text_auto=True  # Automatically add text values
        )

        # Format the text values
        fig.update_traces(
            texttemplate="%{z:.2f}",
            textfont={"size": 10, "color": "white"},
            hovertemplate=(
                f"<b>{x_col}</b>: %{{x}}<br>" +
                f"<b>{y_col}</b>: %{{y}}<br>" +
                f"<b>{value_col}</b>: %{{z:.2f}}<br>" +
                "<extra></extra>"
            )
        )

        # Improve layout
        fig.update_layout(
            margin=dict(t=50, b=70, l=20, r=20),
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        # Add summary statistics below the chart
        fig.add_annotation(
            text=f"Average: {avg_value:.2f} | Max: {max_value:.2f} | Min: {min_value:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        return fig

    elif chart_type == "Box Plot":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")

        if not x_col or not y_col:
            return None

        # Create box plot with enhanced value display
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=f'Box Plot of {y_col} by {x_col}',
            points='all',  # Show all points
            notched=True,  # Show confidence interval around median
            hover_data=[y_col]  # Add y_col to hover data for better tooltips
        )

        # Enhance hover information
        fig.update_traces(
            hovertemplate=(
                f"<b>{x_col}</b>: %{{x}}<br>" +
                f"<b>{y_col}</b>: %{{y:.2f}}<br>" +
                "<extra></extra>"
            ),
            # Make points more visible
            marker=dict(
                opacity=0.7,
                size=6
            )
        )

        # Improve layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=50, b=100, l=20, r=20)  # Increased bottom margin for statistics
        )

        # Add comprehensive summary statistics for each category
        all_stats = []
        for category in df[x_col].unique():
            subset = df[df[x_col] == category][y_col]
            if not subset.empty:
                # Calculate detailed statistics
                count = len(subset)
                median = subset.median()
                mean = subset.mean()
                q1 = subset.quantile(0.25)
                q3 = subset.quantile(0.75)
                iqr = q3 - q1
                min_val = subset.min()
                max_val = subset.max()
                std_dev = subset.std()

                # Add annotation above each box
                fig.add_annotation(
                    x=category,
                    y=max_val,
                    text=f"Median: {median:.2f}<br>Mean: {mean:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10,
                    font=dict(size=10),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.3)",
                    borderwidth=1,
                    borderpad=4
                )

                # Store statistics for overall summary
                all_stats.append({
                    'Category': category,
                    'Count': count,
                    'Mean': mean,
                    'Median': median,
                    'StdDev': std_dev,
                    'Min': min_val,
                    'Max': max_val,
                    'IQR': iqr
                })

        # Add overall summary statistics below the chart
        if all_stats:
            # Create a summary table as text
            summary_text = "<br>".join([
                f"<b>{stat['Category']}</b>: Count={stat['Count']}, Mean={stat['Mean']:.2f}, "
                f"Median={stat['Median']:.2f}, StdDev={stat['StdDev']:.2f}, "
                f"Min={stat['Min']:.2f}, Max={stat['Max']:.2f}"
                for stat in all_stats
            ])

            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=-0.2,
                text=summary_text,
                showarrow=False,
                font=dict(size=10, color="black"),
                align="center",
                bgcolor="rgba(240, 240, 240, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=5
            )

        return fig

    elif chart_type == "Stacked Bar Chart":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")
        color_col = chart_options.get("color_col")

        if not x_col or not y_col or not color_col:
            return None

        # Aggregate data
        agg_df = df.groupby([x_col, color_col])[y_col].sum().reset_index()

        # Calculate statistics for summary
        total_value = agg_df[y_col].sum()
        avg_value = agg_df[y_col].mean()

        # Calculate totals per x category for annotations
        totals_per_category = agg_df.groupby(x_col)[y_col].sum().reset_index()

        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f'{y_col} by {x_col} and {color_col}',
            barmode='stack',
            text=y_col,  # Add value labels
            custom_data=[agg_df[y_col]]  # Add custom data for enhanced hover
        )

        # Format the text labels with improved visibility
        fig.update_traces(
            texttemplate='%{y:.1f}',
            textposition='inside',
            insidetextfont=dict(color='white', size=10, family="Arial"),
            hovertemplate=(
                f"<b>{x_col}</b>: %{{x}}<br>" +
                f"<b>{color_col}</b>: %{{fullData.name}}<br>" +
                f"<b>{y_col}</b>: %{{y:.2f}}<br>" +
                f"<b>Percentage of Total</b>: %{{customdata[0] / {total_value} * 100:.1f}}%<br>" +
                "<extra></extra>"
            )
        )

        # Add total value annotations at the top of each stacked bar
        for i, row in totals_per_category.iterrows():
            fig.add_annotation(
                x=row[x_col],
                y=row[y_col],
                text=f"Total: {row[y_col]:.1f}",
                showarrow=False,
                yshift=10,
                font=dict(size=11, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=2
            )

        # Improve layout with more information
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title=color_col,
            margin=dict(t=50, b=100, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )

        # Add summary statistics below the chart
        fig.add_annotation(
            text=f"Total: {total_value:.2f} | Average per Category: {avg_value:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="gray")
        )

        # Ensure x-axis labels are properly aligned
        if len(df[x_col].unique()) > 5:
            fig.update_xaxes(tickangle=-45)

        return fig

    return None

def show_smart_visualizations(df, numeric_cols, categorical_cols):
    """
    Show smart visualizations with automatic column detection.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names
    """
    st.subheader("Smart Visualizations")

    st.markdown("""
    This feature intelligently selects the most appropriate columns for your chosen chart type.
    Simply select a chart type, and the system will recommend the best columns for that visualization.
    You can always adjust the selections if needed.
    """)

    # Filter chart types based on available data
    available_chart_types = list(SMART_CHART_TYPES)

    # Only show Scatter Plot if we have at least 2 numeric columns
    if len(numeric_cols) < 2:
        if "Scatter Plot" in available_chart_types:
            available_chart_types.remove("Scatter Plot")

    # Only show Heatmap if we have at least 2 categorical columns and 1 numeric column
    if len(categorical_cols) < 2 or len(numeric_cols) < 1:
        if "Heatmap" in available_chart_types:
            available_chart_types.remove("Heatmap")

    # Only show Stacked Bar Chart if we have at least 2 categorical columns and 1 numeric column
    if len(categorical_cols) < 2 or len(numeric_cols) < 1:
        if "Stacked Bar Chart" in available_chart_types:
            available_chart_types.remove("Stacked Bar Chart")

    # Only show Box Plot if we have at least 1 categorical column and 1 numeric column
    if len(categorical_cols) < 1 or len(numeric_cols) < 1:
        if "Box Plot" in available_chart_types:
            available_chart_types.remove("Box Plot")

    # Only show Pie Chart if we have at least 1 categorical column and 1 numeric column
    if len(categorical_cols) < 1 or len(numeric_cols) < 1:
        if "Pie Chart" in available_chart_types:
            available_chart_types.remove("Pie Chart")

    # Only show Time Series if we have date columns
    date_cols = detect_date_columns(df)
    if not date_cols:
        if "Time Series" in available_chart_types:
            available_chart_types.remove("Time Series")

    # Select chart type
    chart_type = st.selectbox(
        "Select Chart Type",
        available_chart_types,
        help="Choose the type of chart you want to create"
    )

    # Get recommended columns for the selected chart type
    recommendations = detect_best_columns_for_chart(df, chart_type, numeric_cols, categorical_cols)

    # Display options based on chart type
    chart_options = {}

    # Create a container for the chart
    chart_container = st.container()

    # Create columns for the options
    col1, col2 = st.columns(2)

    with col1:
        if chart_type == "Pie Chart":
            st.subheader("Chart Options")

            # Category column selection
            category_options = categorical_cols
            default_idx = category_options.index(recommendations["category_col"]) if recommendations.get("category_col") in category_options else 0
            chart_options["category_col"] = st.selectbox(
                "Category Column",
                category_options,
                index=default_idx if category_options else 0,
                help="Column to use for pie chart categories"
            )

            # Value column selection
            value_options = numeric_cols
            default_idx = value_options.index(recommendations["value_col"]) if recommendations.get("value_col") in value_options else 0
            chart_options["value_col"] = st.selectbox(
                "Value Column",
                value_options,
                index=default_idx if value_options else 0,
                help="Column to use for pie chart values"
            )

        elif chart_type == "Bar Chart":
            st.subheader("Chart Options")

            # X-axis column selection
            x_options = categorical_cols
            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "X-Axis Column",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for bar chart categories (x-axis)"
            )

            # Y-axis column selection
            y_options = numeric_cols
            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Y-Axis Column",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for bar chart values (y-axis)"
            )



    with col2:
        if chart_type == "Time Series":
            st.subheader("Chart Options")

            # X-axis column selection
            if chart_type == "Time Series":
                x_options = detect_date_columns(df)
                if not x_options:
                    st.warning("No date columns detected. Please select a different chart type or ensure your data contains date columns.")
                    return
            else:
                x_options = numeric_cols + categorical_cols

            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "X-Axis Column",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for x-axis"
            )

            # Y-axis column selection
            y_options = numeric_cols
            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Y-Axis Column",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for y-axis"
            )



            # Data aggregation options for time series
            if chart_type == "Time Series" or (chart_type == "Line Chart" and len(df) > 5000):
                st.markdown("### Data Aggregation")

                # Check if we're dealing with a large dataset
                is_large_dataset = len(df) > 5000

                # Show info message for large datasets
                if is_large_dataset:
                    st.info(
                        "This dataset contains a large number of points. "
                        "Automatic data aggregation will be applied to improve performance and readability."
                    )

                # Aggregation level selection
                aggregation_options = [
                    ("auto", "Automatic (recommended)"),
                    ("none", "None (raw data)"),
                    ("day", "Daily"),
                    ("week", "Weekly"),
                    ("month", "Monthly"),
                    ("quarter", "Quarterly"),
                    ("year", "Yearly")
                ]

                # Add a note about raw data option
                if len(df) > 1000:
                    st.info("Note: The 'None (raw data)' option may cause performance issues with large datasets.")
                    st.markdown("For best results with large datasets, use 'Automatic' or select a specific aggregation level.")

                # Default to automatic for large datasets
                default_aggregation = "auto" if is_large_dataset else "none"

                # Create a radio button for aggregation selection
                selected_aggregation = st.radio(
                    "Time Aggregation Level",
                    options=[option[0] for option in aggregation_options],
                    format_func=lambda x: dict(aggregation_options)[x],
                    index=[option[0] for option in aggregation_options].index(default_aggregation),
                    help="Select how to aggregate data points over time. 'Automatic' will choose the best level based on your data."
                )

                chart_options["aggregation"] = selected_aggregation

        elif chart_type == "Scatter Plot":
            st.subheader("Chart Options")

            # X-axis column selection
            x_options = numeric_cols
            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "X-Axis Column",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for x-axis"
            )

            # Y-axis column selection
            y_options = [col for col in numeric_cols if col != chart_options["x_col"]]
            if not y_options and numeric_cols:
                y_options = numeric_cols  # If only one numeric column, use it for both axes

            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Y-Axis Column",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for y-axis"
            )



            # Size column selection (optional)
            use_size = st.checkbox("Use Size Variation", value=recommendations.get("size_col") is not None)
            if use_size and len(numeric_cols) > 2:
                size_options = [col for col in numeric_cols if col != chart_options["x_col"] and col != chart_options["y_col"]]
                if size_options:
                    default_idx = size_options.index(recommendations["size_col"]) if recommendations.get("size_col") in size_options else 0
                    chart_options["size_col"] = st.selectbox(
                        "Size Column",
                        size_options,
                        index=default_idx,
                        help="Column to use for point sizes"
                    )

        elif chart_type == "Heatmap":
            st.subheader("Chart Options")

            # X-axis column selection
            x_options = categorical_cols
            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "X-Axis Column",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for heatmap x-axis"
            )

            # Y-axis column selection
            y_options = [col for col in categorical_cols if col != chart_options["x_col"]]
            if not y_options and len(categorical_cols) > 0:
                y_options = categorical_cols  # If only one categorical column, use it for both axes

            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Y-Axis Column",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for heatmap y-axis"
            )

            # Value column selection
            value_options = numeric_cols
            default_idx = value_options.index(recommendations["value_col"]) if recommendations.get("value_col") in value_options else 0
            chart_options["value_col"] = st.selectbox(
                "Value Column",
                value_options,
                index=default_idx if value_options else 0,
                help="Column to use for heatmap values (color intensity)"
            )

        elif chart_type == "Box Plot":
            st.subheader("Chart Options")

            # X-axis column selection (categorical)
            x_options = categorical_cols
            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "Category Column (X-Axis)",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for box plot categories"
            )

            # Y-axis column selection (numeric)
            y_options = numeric_cols
            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Value Column (Y-Axis)",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for box plot values"
            )

        elif chart_type == "Stacked Bar Chart":
            st.subheader("Chart Options")

            # X-axis column selection
            x_options = categorical_cols
            default_idx = x_options.index(recommendations["x_col"]) if recommendations.get("x_col") in x_options else 0
            chart_options["x_col"] = st.selectbox(
                "X-Axis Column",
                x_options,
                index=default_idx if x_options else 0,
                help="Column to use for bar chart categories (x-axis)"
            )

            # Color column selection
            color_options = [col for col in categorical_cols if col != chart_options["x_col"]]
            if color_options:
                default_idx = color_options.index(recommendations["color_col"]) if recommendations.get("color_col") in color_options else 0
                chart_options["color_col"] = st.selectbox(
                    "Color/Stack Column",
                    color_options,
                    index=default_idx,
                    help="Column to use for stacking bars"
                )
            else:
                st.warning("Stacked bar charts require at least 2 categorical columns. Please select a different chart type.")
                return

            # Y-axis column selection
            y_options = numeric_cols
            default_idx = y_options.index(recommendations["y_col"]) if recommendations.get("y_col") in y_options else 0
            chart_options["y_col"] = st.selectbox(
                "Y-Axis Column",
                y_options,
                index=default_idx if y_options else 0,
                help="Column to use for bar chart values (y-axis)"
            )

    # Create and display the chart
    with chart_container:
        try:
            fig = create_smart_chart(df, chart_type, chart_options)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Add chart explanation
                st.markdown("### Chart Insights")
            else:
                st.error("Could not create chart with the selected options. Please try different columns or chart type.")
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Try selecting different columns or a different chart type. If the issue persists with Time Series charts, check that your date column contains valid dates.")
            # Create a simple fallback message
            st.markdown("### Troubleshooting Tips")
            st.markdown("""
            For Time Series charts:
            - Ensure your date column contains valid dates
            - Try a different aggregation level (e.g., Monthly instead of Automatic)
            - Check for missing or invalid values in your data
            """)
            # Set fig to None so we don't try to access it later
            fig = None

        # Only proceed with insights if we have a valid figure
        if fig:
            if chart_type == "Pie Chart":
                st.markdown(f"""
                This pie chart shows the distribution of **{chart_options.get('value_col')}** across different
                **{chart_options.get('category_col')}** categories. Each slice represents the proportion of the total.
                """)
            elif chart_type == "Bar Chart":
                st.markdown(f"""
                This bar chart compares **{chart_options.get('y_col')}** across different
                **{chart_options.get('x_col')}** categories.
                """)

            elif chart_type == "Time Series":
                # Base description
                description = f"""
                This time series chart shows how **{chart_options.get('y_col')}** changes over time
                (**{chart_options.get('x_col')}**).
                """

                # Add information about data aggregation if applied
                # Check if aggregation info exists in the figure's layout metadata
                # Use a try-except block to handle cases where layout properties might not exist
                try:
                    if hasattr(fig, 'layout') and hasattr(fig.layout, 'get'):
                        # Try to get the meta_aggregation_info property
                        agg_info = fig.layout.get('meta_aggregation_info', {})
                        if isinstance(agg_info, dict) and agg_info.get('applied', False):
                            aggregation_message = agg_info.get('message', '')
                            if aggregation_message:
                                description += f"""

                                **Data Handling:** {aggregation_message}
                                """
                except Exception as e:
                    # If accessing the layout properties fails, just continue without the aggregation message
                    # st.warning(f"Could not access aggregation info: {str(e)}")
                    pass

                st.markdown(description)
            elif chart_type == "Scatter Plot":
                desc = f"""
                This scatter plot shows the relationship between **{chart_options.get('x_col')}** and
                **{chart_options.get('y_col')}**
                """
                if "size_col" in chart_options:
                    desc += f", with points sized by **{chart_options.get('size_col')}**"
                desc += "."
                st.markdown(desc)
            elif chart_type == "Heatmap":
                st.markdown(f"""
                This heatmap shows the relationship between **{chart_options.get('x_col')}** (x-axis) and
                **{chart_options.get('y_col')}** (y-axis), with color intensity representing the value of
                **{chart_options.get('value_col')}**. Darker colors indicate higher values.
                """)
            elif chart_type == "Box Plot":
                st.markdown(f"""
                This box plot shows the distribution of **{chart_options.get('y_col')}** across different
                **{chart_options.get('x_col')}** categories. The box represents the interquartile range (IQR),
                the line inside the box is the median, and the whiskers extend to the minimum and maximum values
                (excluding outliers).
                """)
            elif chart_type == "Stacked Bar Chart":
                st.markdown(f"""
                This stacked bar chart shows **{chart_options.get('y_col')}** across different
                **{chart_options.get('x_col')}** categories, with each bar divided into segments representing
                **{chart_options.get('color_col')}**. The height of each segment represents its contribution to the total.
                """)
        else:
            st.error("Could not create chart with the selected options. Please try different columns or chart type.")
