"""
Smart visualization components for the application.
This module provides intelligent chart selection and automatic column detection.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
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
        )

        # Improve layout
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            margin=dict(t=50, b=50, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )

        return fig

    elif chart_type == "Bar Chart":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")

        if not x_col or not y_col:
            return None

        # Aggregate data - no color grouping
        agg_df = df.groupby(x_col)[y_col].sum().reset_index()
        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            title=f'{y_col} by {x_col}',
            text=y_col  # Add value labels
        )

        # Format the text labels
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='outside'
        )

        # Improve layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=50, b=100, l=20, r=20),
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
                    optimal_aggregation, aggregation_message = determine_optimal_aggregation(df_sorted, x_col)

                    # Apply aggregation if needed
                    if optimal_aggregation != 'none':
                        # Group by time period and calculate mean
                        if optimal_aggregation == 'day':
                            df_sorted = df_sorted.set_index(x_col).resample('D').mean().reset_index()
                        elif optimal_aggregation == 'week':
                            df_sorted = df_sorted.set_index(x_col).resample('W').mean().reset_index()
                        elif optimal_aggregation == 'month':
                            df_sorted = df_sorted.set_index(x_col).resample('M').mean().reset_index()
                        elif optimal_aggregation == 'quarter':
                            df_sorted = df_sorted.set_index(x_col).resample('Q').mean().reset_index()
                        elif optimal_aggregation == 'year':
                            df_sorted = df_sorted.set_index(x_col).resample('Y').mean().reset_index()

                        aggregation_applied = True
                # If user selected a specific aggregation level
                elif user_aggregation != 'none':
                    # Apply user-selected aggregation
                    if user_aggregation == 'day':
                        df_sorted = df_sorted.set_index(x_col).resample('D').mean().reset_index()
                    elif user_aggregation == 'week':
                        df_sorted = df_sorted.set_index(x_col).resample('W').mean().reset_index()
                    elif user_aggregation == 'month':
                        df_sorted = df_sorted.set_index(x_col).resample('M').mean().reset_index()
                    elif user_aggregation == 'quarter':
                        df_sorted = df_sorted.set_index(x_col).resample('Q').mean().reset_index()
                    elif user_aggregation == 'year':
                        df_sorted = df_sorted.set_index(x_col).resample('Y').mean().reset_index()

                    aggregation_applied = True
                    aggregation_message = f"Data aggregated to {user_aggregation}ly level as selected"

        # Determine if we should add markers based on data size
        use_markers = len(df_sorted) < 100

        # Create the chart - no color grouping
        fig = px.line(
            df_sorted,
            x=x_col,
            y=y_col,
            title=f'Trend of {y_col} over {x_col}',
            markers=use_markers
        )

        # Improve layout with more space at the top to avoid overlapping with range selector buttons
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=70, b=50, l=20, r=20),  # Increased top margin to accommodate range selector
            hovermode="closest"
        )

        # Add annotation about data aggregation if applied - positioned to avoid overlap with range selector
        if aggregation_applied and aggregation_message:
            # Add as a subtitle instead of an annotation to avoid overlap with range selector buttons
            title = fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else f'Trend of {y_col} over {x_col}'
            subtitle = f"<br><span style='font-size:10px;color:gray'>{aggregation_message}</span>"
            fig.update_layout(
                title={
                    'text': title + subtitle,
                    'y': 0.95,  # Position the title a bit higher
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                }
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
            fig.update_layout(
                _aggregation_info={
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
            # No color grouping - removed as requested
            if size_col_to_use:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    size=size_col_to_use,
                    title=f'{y_col} vs {x_col} (size: {size_col})',
                    opacity=0.7
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=x_col,
                    y=y_col,
                    title=f'{y_col} vs {x_col}',
                    opacity=0.7
                )

            # Add trendline
            fig.update_layout(
                xaxis_title=x_col,
                yaxis_title=y_col,
                margin=dict(t=50, b=50, l=20, r=20),
                hovermode="closest"
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

        # Create heatmap
        fig = px.imshow(
            pivot_df,
            labels=dict(x=x_col, y=y_col, color=value_col),
            x=pivot_df.columns,
            y=pivot_df.index,
            title=f'Heatmap of {value_col} by {x_col} and {y_col}',
            color_continuous_scale='Viridis'
        )

        # Improve layout
        fig.update_layout(
            margin=dict(t=50, b=50, l=20, r=20),
            xaxis_title=x_col,
            yaxis_title=y_col
        )

        return fig

    elif chart_type == "Box Plot":
        x_col = chart_options.get("x_col")
        y_col = chart_options.get("y_col")

        if not x_col or not y_col:
            return None

        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=f'Box Plot of {y_col} by {x_col}',
            points='all',  # Show all points
            notched=True   # Show confidence interval around median
        )

        # Improve layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            margin=dict(t=50, b=50, l=20, r=20)
        )

        # Add summary statistics
        for category in df[x_col].unique():
            subset = df[df[x_col] == category][y_col]
            if not subset.empty:
                median = subset.median()
                mean = subset.mean()
                fig.add_annotation(
                    x=category,
                    y=subset.max(),
                    text=f"Mean: {mean:.2f}<br>Median: {median:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
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

        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=f'{y_col} by {x_col} and {color_col}',
            barmode='stack',
            text=y_col  # Add value labels
        )

        # Format the text labels
        fig.update_traces(
            texttemplate='%{y:.2f}',
            textposition='inside',
            insidetextfont=dict(color='white', size=10)
        )

        # Improve layout
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title=color_col,
            margin=dict(t=50, b=100, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
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
        fig = create_smart_chart(df, chart_type, chart_options)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

            # Add chart explanation
            st.markdown("### Chart Insights")
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
                if fig.layout.get('_aggregation_info', {}).get('applied', False):
                    aggregation_message = fig.layout.get('_aggregation_info', {}).get('message', '')
                    if aggregation_message:
                        description += f"""

                        **Data Handling:** {aggregation_message}
                        """

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
