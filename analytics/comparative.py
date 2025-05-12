"""
Comparative analysis components for the application.
"""
import streamlit as st
from utils.data_utils import detect_date_columns, create_download_link
from visualizations.chart_utils import create_comparative_chart

def show_comparative_analysis(df, numeric_cols, categorical_cols):
    """Show comparative analysis of multiple metrics."""
    st.write("### Comparative Analysis")

    if numeric_cols:
        # Select columns for comparison
        y_cols = st.multiselect("Select metrics to compare", numeric_cols,
                               default=numeric_cols[:min(3, len(numeric_cols))])

        if y_cols and len(y_cols) >= 2:
            # Select X-axis column
            x_options = categorical_cols + detect_date_columns(df)
            if x_options:
                x_col = st.selectbox("Select X-axis for comparison", x_options)

                # Select chart type
                chart_type = st.radio("Select chart type", ["bar", "line"], horizontal=True)

                # Add aggregation method selector
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("**Aggregation Method:**")
                with col2:
                    aggregation_method = st.radio(
                        "Select aggregation method",
                        ["mean", "sum", "min", "max"],
                        horizontal=True,
                        label_visibility="collapsed",
                        index=1 if "sum" in ["mean", "sum", "min", "max"] else 0  # Default to sum if available
                    )

                # Create comparative chart with selected aggregation method
                comp_fig = create_comparative_chart(df, x_col, y_cols, chart_type, aggregation_method)
                st.plotly_chart(comp_fig, use_container_width=True)

                # Add chart summary and key insights
                with st.container():
                    st.markdown("""
                    <div style="background-color:#f0f2f6; padding:15px; border-radius:5px; margin-bottom:20px;">
                        <h4 style="color:#0e1117; margin-top:0;">📊 Chart Summary</h4>
                        <ul style="margin-bottom:0;">
                    """, unsafe_allow_html=True)

                    # Generate insights based on the data
                    summary_points = []

                    # Get aggregated data for insights using the selected aggregation method
                    agg_data = df.groupby(x_col)[y_cols].agg([aggregation_method, 'max', 'min']).reset_index()
                    agg_method_display = aggregation_method.upper()

                    # Add insight about highest values
                    for col in y_cols:
                        max_category = agg_data.loc[agg_data[(col, aggregation_method)].idxmax()][x_col]
                        max_value = agg_data[(col, aggregation_method)].max()
                        summary_points.append(f"<li><strong>{col}</strong> reaches its highest {aggregation_method} value of <strong>{max_value:.2f}</strong> in <strong>{max_category}</strong>.</li>")

                    # Add insight about comparison between categories if there are multiple
                    if len(agg_data) > 1:
                        for col in y_cols:
                            max_idx = agg_data[(col, aggregation_method)].idxmax()
                            min_idx = agg_data[(col, aggregation_method)].idxmin()
                            if max_idx != min_idx:  # Only if they're different
                                max_category = agg_data.loc[max_idx][x_col]
                                min_category = agg_data.loc[min_idx][x_col]
                                max_value = agg_data.loc[max_idx][(col, aggregation_method)]
                                min_value = agg_data.loc[min_idx][(col, aggregation_method)]
                                diff_pct = ((max_value - min_value) / min_value * 100) if min_value != 0 else 0
                                summary_points.append(f"<li><strong>{col}</strong> is <strong>{diff_pct:.1f}%</strong> higher in <strong>{max_category}</strong> compared to <strong>{min_category}</strong> (using {agg_method_display}).</li>")

                    # Add overall trend insight if using a date column
                    date_cols = detect_date_columns(df)
                    if x_col in date_cols and len(agg_data) > 2:
                        for col in y_cols:
                            first_val = agg_data.iloc[0][(col, aggregation_method)]
                            last_val = agg_data.iloc[-1][(col, aggregation_method)]
                            if first_val != 0:
                                change_pct = ((last_val - first_val) / first_val * 100)
                                trend = "increased" if change_pct > 0 else "decreased"
                                summary_points.append(f"<li>Over the time period, <strong>{col}</strong> has <strong>{trend} by {abs(change_pct):.1f}%</strong> (using {agg_method_display}).</li>")

                    # Display all insights
                    for point in summary_points[:5]:  # Limit to 5 insights
                        st.markdown(point, unsafe_allow_html=True)

                    st.markdown("</ul></div>", unsafe_allow_html=True)

                # Add year-over-year or category-wise comparison table
                st.write("### Comparison Table")

                # Group by the X column and calculate statistics for each Y column
                comparison_df = df.groupby(x_col)[y_cols].agg(['sum', 'mean', 'min', 'max', 'std']).reset_index()

                # Add a note about the selected aggregation method
                st.info(f"The chart above is using the **{aggregation_method.upper()}** aggregation method. The table below shows all available statistics.")

                # Apply styling to improve readability
                st.dataframe(
                    comparison_df,
                    column_config={
                        x_col: st.column_config.TextColumn(
                            x_col,
                            width="medium",
                            help=f"Values of {x_col}"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )

                # Download option
                if st.button("Download Comparison Data"):
                    st.markdown(create_download_link(comparison_df, "comparison_data.csv",
                                                   "Click here to download the comparison data"),
                               unsafe_allow_html=True)
            else:
                st.info("No suitable X-axis columns found for comparison.")
        else:
            st.info("Please select at least two metrics for comparison.")
    else:
        st.info("No numeric columns found for comparative analysis.")
