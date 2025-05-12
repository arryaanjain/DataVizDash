"""
Advanced visualization components for the application.
"""
import streamlit as st
import pandas as pd
from utils.data_utils import detect_date_columns, create_download_link
from visualizations.chart_utils import (
    create_pie_chart, create_area_chart,
    create_box_plot, create_stacked_bar, create_enhanced_time_series_chart
)
from config import RICH_VIZ_TYPES

def show_rich_visualizations(df, numeric_cols, categorical_cols):
    """Show rich graphical visuals."""
    st.write("### Rich Graphical Visuals")

    # Select visualization type
    viz_type = st.selectbox(
        "Select Visualization Type",
        RICH_VIZ_TYPES
    )

    if viz_type == "Pie Chart":
        if categorical_cols:
            pie_col = st.selectbox("Select column for pie chart", categorical_cols)
            pie_fig = create_pie_chart(df, pie_col)
            st.plotly_chart(pie_fig, use_container_width=True)

            # Add explanatory key points below the chart
            with st.expander("About This Visualization", expanded=True):
                st.markdown("""
                ### Pie Chart Interpretation

                This pie chart shows the distribution of values in the selected categorical column. Each slice represents a category, with the size proportional to its frequency in the dataset.

                **Key Points:**
                - Larger slices represent more frequent categories
                - Percentages show the relative proportion of each category
                - The 'Total' value shows the number of data points represented
                - If there are more than 5 categories, only the top 5 are shown individually, with the rest grouped as 'Others'

                **Use this visualization to:**
                - Understand the composition of your categorical data
                - Identify dominant categories
                - Compare the relative sizes of different categories
                """)
        else:
            st.info("No categorical columns found for pie chart.")



    elif viz_type == "Enhanced Time Series":
        if numeric_cols:
            date_cols = detect_date_columns(df)
            if date_cols:
                st.write("### Enhanced Time Series Visualization")
                st.write("This visualization is optimized for dense time series data with advanced filtering and aggregation options.")

                # Column selection
                x_col = st.selectbox("Select X-axis (date) column", date_cols)
                y_col = st.selectbox("Select Y-axis column for time series", numeric_cols)

                # Date range filtering
                st.write("### Date Range Filtering")

                # Get min and max dates from the dataset
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                    min_date = df[x_col].min().date()
                    max_date = df[x_col].max().date()

                    # Extract years for year filter
                    years = sorted(df[x_col].dt.year.unique())

                    # Year filter
                    col1, col2 = st.columns(2)

                    with col1:
                        selected_year = st.selectbox(
                            "Filter by Year",
                            ["All Years"] + [str(year) for year in years],
                            help="Select a specific year to view"
                        )

                    with col2:
                        use_custom_range = st.checkbox(
                            "Use Custom Date Range",
                            value=False,
                            help="Specify a custom date range instead of filtering by year"
                        )

                    # Date range
                    if use_custom_range:
                        date_range_col1, date_range_col2 = st.columns(2)
                        with date_range_col1:
                            start_date = st.date_input(
                                "Start Date",
                                value=min_date,
                                min_value=min_date,
                                max_value=max_date
                            )
                        with date_range_col2:
                            end_date = st.date_input(
                                "End Date",
                                value=max_date,
                                min_value=min_date,
                                max_value=max_date
                            )

                        # Create date range tuple
                        date_range = (start_date, end_date)
                    else:
                        # If year filter is active
                        if selected_year != "All Years":
                            year = int(selected_year)
                            start_date = pd.Timestamp(year=year, month=1, day=1)
                            end_date = pd.Timestamp(year=year, month=12, day=31)
                            date_range = (start_date, end_date)
                        else:
                            date_range = None
                except:
                    st.warning("Could not determine date range from the selected date column.")
                    date_range = None

                # Create columns for visualization options
                st.write("### Visualization Options")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Time aggregation options
                    aggregation = st.selectbox(
                        "Time Aggregation",
                        ["none", "day", "week", "month", "quarter", "year"],
                        help="Aggregate data points by time period to reduce noise"
                    )

                with col2:
                    # Rolling average / smoothing
                    use_rolling_avg = st.checkbox(
                        "Apply Rolling Average",
                        value=False,  # Default to OFF as per user request
                        help="Smooth the line by calculating moving averages"
                    )

                    rolling_window = None
                    if use_rolling_avg:
                        rolling_window = st.slider(
                            "Window Size",
                            min_value=2,
                            max_value=50,
                            value=14,
                            help="Number of data points to include in the rolling average"
                        )

                with col3:
                    # Outlier removal
                    remove_outliers = st.checkbox(
                        "Remove Outliers",
                        value=False,  # Default to OFF for simplicity
                        help="Remove extreme values that might skew the visualization"
                    )

                    outlier_threshold = 3.0
                    if remove_outliers:
                        outlier_threshold = st.slider(
                            "Outlier Z-Score Threshold",
                            min_value=1.0,
                            max_value=5.0,
                            value=3.0,
                            step=0.1,
                            help="Z-score threshold for outlier detection (lower = more aggressive filtering)"
                        )

                # No color grouping - removed as requested
                color_col = None
                selected_categories = None

                # Create the enhanced time series chart
                ts_fig = create_enhanced_time_series_chart(
                    df,
                    x_col,
                    y_col,
                    color_col=color_col,
                    aggregation=aggregation,
                    rolling_window=rolling_window if use_rolling_avg else None,
                    remove_outliers=remove_outliers,
                    outlier_threshold=outlier_threshold,
                    date_range=date_range,
                    selected_categories=selected_categories
                )

                st.plotly_chart(ts_fig, use_container_width=True)

                # Add information about the visualization
                with st.expander("Visualization Details", expanded=True):
                    # Format date range info for display
                    date_range_info = "All available dates"
                    if date_range is not None:
                        start_date, end_date = date_range
                        if start_date is not None and end_date is not None:
                            date_range_info = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

                    # No category selection info needed since color grouping is removed

                    st.markdown(
                        "### Enhanced Time Series Interpretation\n\n" +
                        f"This time series visualization shows the trend of **{y_col}** over time (**{x_col}**). " +
                        f"The chart has been optimized for clarity and insight with value labels at key points.\n\n" +
                        "**Key Features:**\n\n" +
                        "- **Date Range Filtering**: Focus on specific time periods\n" +
                        "- **Time Aggregation**: Reduces data density by grouping points by time period\n" +
                        "- **Rolling Average**: Smooths the line to reveal underlying trends\n" +
                        "- **Outlier Removal**: Filters extreme values that can distort the visualization\n" +
                        "- **Interactive Time Navigation**: Use the range slider below the chart to zoom in on specific time periods\n" +
                        "- **Value Labels**: Key points (minimum, maximum, and other significant values) are labeled directly on the chart\n\n" +
                        "**How to Interpret This Chart:**\n\n" +
                        "- The line shows how values change over time\n" +
                        "- Upward slopes indicate increasing values, downward slopes indicate decreasing values\n" +
                        "- Steeper slopes indicate more rapid changes\n" +
                        "- If rolling average is enabled, the smoothed line helps identify the underlying trend\n" +
                        "- Value labels highlight key points of interest\n\n" +
                        "**Current Settings:**\n" +
                        f"- Date range: **{date_range_info}**\n" +
                        f"- Data aggregation: **{aggregation}**\n" +
                        (f"- Rolling average window: **{rolling_window} points**\n" if use_rolling_avg else "- Rolling average: **disabled**\n") +
                        (f"- Outlier removal: **enabled** (z-score > {outlier_threshold})\n" if remove_outliers else "- Outlier removal: **disabled**\n") +
                        "\n**Use this visualization to:**\n" +
                        "- Identify trends and patterns over time\n" +
                        "- Detect seasonal patterns or cycles\n" +
                        "- Spot anomalies or unusual events\n" +
                        "- Forecast future values based on historical patterns\n" +
                        "- Compare performance across different time periods"
                    )
            else:
                st.warning("Enhanced Time Series visualization requires date columns. No date columns were detected in your data.")
        else:
            st.info("No numeric columns found for time series visualization.")

    elif viz_type == "Area Chart":
        if numeric_cols:
            date_cols = detect_date_columns(df)
            if date_cols:
                x_col = st.selectbox("Select X-axis (date) column", date_cols)
                y_col = st.selectbox("Select Y-axis column for area chart", numeric_cols)

                # Create area chart
                area_fig = create_area_chart(df, x_col, y_col)
                st.plotly_chart(area_fig, use_container_width=True)

                # Add explanatory key points below the chart
                with st.expander("About This Visualization", expanded=True):
                    st.markdown(f"""
                    ### Area Chart Interpretation

                    This area chart shows the trend of **{y_col}** over **{x_col}**. The filled area emphasizes the magnitude of values and how they change over time.

                    **Key Points:**
                    - The height of the area represents the value at each point in time
                    - Value labels are shown at key points (minimum, maximum, and last point)
                    - Summary statistics (average, maximum, minimum) are displayed below the chart
                    - Smooth curves are used to emphasize the overall trend rather than individual fluctuations

                    **Use this visualization to:**
                    - Identify trends and patterns over time
                    - Spot periods of significant increase or decrease
                    - Understand the overall magnitude and range of values
                    - Compare values across different time periods
                    """)
            else:
                st.warning("No date columns detected for area chart. Please select columns manually.")
                x_col = st.selectbox("Select X-axis column", numeric_cols)
                y_col = st.selectbox("Select Y-axis column for area chart",
                                    numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

                area_fig = create_area_chart(df, x_col, y_col)
                st.plotly_chart(area_fig, use_container_width=True)

                # Add explanatory key points below the chart
                with st.expander("About This Visualization", expanded=True):
                    st.markdown(f"""
                    ### Area Chart Interpretation

                    This area chart shows the relationship between **{y_col}** and **{x_col}**. The filled area emphasizes the magnitude of values and how they change across the x-axis.

                    **Key Points:**
                    - The height of the area represents the value at each point
                    - Value labels are shown at key points (minimum, maximum, and last point)
                    - Summary statistics (average, maximum, minimum) are displayed below the chart
                    - Smooth curves are used to emphasize the overall trend rather than individual fluctuations

                    **Use this visualization to:**
                    - Identify trends and patterns
                    - Spot regions of significant increase or decrease
                    - Understand the overall magnitude and range of values
                    - Compare values across different ranges
                    """)
        else:
            st.info("No numeric columns found for area chart.")

    elif viz_type == "Box Plot":
        if numeric_cols and categorical_cols:
            x_col = st.selectbox("Select X-axis (categorical) column", categorical_cols)
            y_col = st.selectbox("Select Y-axis (numeric) column for box plot", numeric_cols)

            box_fig = create_box_plot(df, x_col, y_col)
            st.plotly_chart(box_fig, use_container_width=True)

            # Add explanatory key points below the chart
            with st.expander("About This Visualization", expanded=True):
                st.markdown(f"""
                ### Box Plot Interpretation

                This box plot shows the distribution of **{y_col}** across different categories of **{x_col}**. Each box represents the distribution of values within a category.

                **Key Elements of the Box Plot:**
                - **Box**: The box represents the interquartile range (IQR), from the 25th percentile (Q1) to the 75th percentile (Q3)
                - **Line inside box**: The median (50th percentile)
                - **Notch**: The confidence interval around the median
                - **Whiskers**: Extend to the most extreme data points within 1.5 times the IQR
                - **Points**: Individual data points, with potential outliers shown outside the whiskers

                **Value Labels:**
                - Median values are prominently displayed for each category
                - Mean values are shown to compare with medians
                - Q1 and Q3 values show the range containing the middle 50% of data

                **Use this visualization to:**
                - Compare distributions across different categories
                - Identify differences in central tendency (median/mean)
                - Spot variations in data spread (wider boxes indicate more variability)
                - Detect potential outliers
                - Assess the symmetry or skewness of distributions
                """)
        else:
            st.info("Box plots require both numeric and categorical columns.")

    elif viz_type == "Stacked Bar Chart":
        if numeric_cols and categorical_cols:
            x_col = st.selectbox("Select X-axis column", categorical_cols)
            y_col = st.selectbox("Select Y-axis (numeric) column", numeric_cols)

            if len(categorical_cols) >= 2:
                color_col = st.selectbox("Select column for stacking",
                                        [c for c in categorical_cols if c != x_col])

                # Prepare data for stacked bar chart
                agg_df = df.groupby([x_col, color_col])[y_col].sum().reset_index()

                stacked_fig = create_stacked_bar(agg_df, x_col, y_col, color_col)
                st.plotly_chart(stacked_fig, use_container_width=True)

                # Add explanatory key points below the chart
                with st.expander("About This Visualization", expanded=True):
                    st.markdown(f"""
                    ### Stacked Bar Chart Interpretation

                    This stacked bar chart shows the distribution of **{y_col}** across categories of **{x_col}**, further broken down by **{color_col}**. Each bar represents a category of {x_col}, and the colored segments within each bar represent different values of {color_col}.

                    **Key Points:**
                    - The height of each bar represents the total sum of {y_col} for that {x_col} category
                    - Each colored segment shows the contribution from a specific {color_col} value
                    - Value labels on each segment show the exact value of that segment
                    - The total value across all categories is displayed at the bottom of the chart

                    **Use this visualization to:**
                    - Compare total values across different categories
                    - Analyze the composition of each category
                    - Identify which subcategories contribute most to each main category
                    - Spot patterns in how subcategories are distributed across main categories
                    - Understand both absolute values (segment size) and relative proportions (segment percentage of the whole)
                    """)
            else:
                st.info("Stacked bar charts require at least 2 categorical columns.")
        else:
            st.info("Stacked bar charts require both numeric and categorical columns.")

    # Download option
    if st.button("Download Visualization Data"):
        st.markdown(create_download_link(df, "visualization_data.csv",
                                       "Click here to download the data"), unsafe_allow_html=True)

def show_advanced_visualizations(df, numeric_cols, categorical_cols):
    """Show advanced visualizations using tabs."""
    st.subheader("Data Visualization")

    # Create tabs for different visualization types
    viz_tabs = st.tabs(["Numeric", "Categorical", "Relationships", "Rich Visuals"])

    # Numeric data visualization tab
    with viz_tabs[0]:
        from visualizations.basic_charts import show_numeric_visualization
        show_numeric_visualization(df, numeric_cols)

    # Categorical data visualization tab
    with viz_tabs[1]:
        from visualizations.basic_charts import show_categorical_visualization
        show_categorical_visualization(df, categorical_cols)

    # Relationships visualization tab
    with viz_tabs[2]:
        from visualizations.basic_charts import show_relationship_visualization
        show_relationship_visualization(df, numeric_cols)

    # Rich Visuals tab
    with viz_tabs[3]:
        show_rich_visualizations(df, numeric_cols, categorical_cols)
