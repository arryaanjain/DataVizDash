"""
KPI dashboard components for the application.
"""
import streamlit as st
import pandas as pd
from utils.data_utils import detect_date_columns, standardize_date_column

def create_kpi_card(title, value, delta=None, delta_suffix="vs previous period"):
    """Create a KPI card with a metric and enhanced styling."""
    # Create a container for the KPI card with a border and background
    with st.container():
        # Apply custom styling with HTML
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; border: 1px solid #eaecf0;">
            <h4 style="margin-top: 0; color: #1f2937; font-size: 16px;">{title}</h4>
        </div>
        """, unsafe_allow_html=True)

        # Add the metric with the value and delta
        st.metric(
            label="",  # Empty label since we added it in the HTML above
            value=value,
            delta=delta,
            delta_color="normal",
            help=f"{title} {delta_suffix}" if delta else None,
        )

        # Add the comparison period as a small caption
        if delta:
            st.caption(f"{delta_suffix}")

def show_kpi_dashboard(df, numeric_cols):
    """Show KPI dashboard with key metrics."""
    st.write("### Key Performance Indicators")

    if numeric_cols:
        # Create a container for the controls
        with st.container():
            col1, col2 = st.columns([3, 1])

            # Select columns for KPIs
            with col1:
                kpi_cols = st.multiselect("Select columns for KPIs", numeric_cols,
                                         default=numeric_cols[:min(4, len(numeric_cols))])

            # Time frame selector for comparison
            with col2:
                time_frame = st.selectbox(
                    "Comparison Period",
                    ["Half-Half", "Monthly", "Quarterly", "Yearly"],
                    help="Select the time period for KPI comparison"
                )

                # Show a small explanation based on the selected time frame
                if time_frame == "Half-Half":
                    st.caption("Compares first half vs second half of data")
                elif time_frame == "Monthly":
                    st.caption("Compares current month vs previous month")
                elif time_frame == "Quarterly":
                    st.caption("Compares current quarter vs previous quarter")
                elif time_frame == "Yearly":
                    st.caption("Compares current year vs previous year")

        # Get date columns (only those with datetime data type)
        date_cols = detect_date_columns(df)

        # Check for standardized date columns (created for columns with hidden year components)
        standardized_cols = [col for col in df.columns if col.endswith('_standardized') and col.replace('_standardized', '') in date_cols]

        # Add standardized columns to date_cols if they exist
        if standardized_cols:
            # Add them at the beginning of the list to prioritize them
            date_cols = standardized_cols + [col for col in date_cols if f"{col}_standardized" not in standardized_cols]

            # Show info message about standardized columns
            st.info("📅 Standardized date columns are available and recommended for year-over-year comparisons.")

        # Initialize selected_date_col at a higher scope so it's accessible throughout the function
        selected_date_col = None

        # Add date column selector if date columns are available
        if date_cols:
            # Create a dropdown to select the date column
            selected_date_col = st.selectbox(
                "Select date column for comparison",
                date_cols,
                index=0,
                help="Select which date column to use for dividing data into time periods"
            )

            # If a standardized column exists for the selected column, suggest using it for yearly comparisons
            if time_frame == "Yearly" and selected_date_col in date_cols and f"{selected_date_col}_standardized" in df.columns:
                st.warning(f"⚠️ For yearly comparisons with '{selected_date_col}', consider using '{selected_date_col}_standardized' instead for more accurate results.")
        else:
            # Display a message if no valid date columns are found
            st.warning("⚠️ No valid date columns found with complete date format (day, month, and year). KPI comparisons will be limited to simple half-half splits without date context.")

        if kpi_cols:
            # Create KPI cards in a grid
            cols = st.columns(len(kpi_cols))

            for i, col in enumerate(kpi_cols):
                with cols[i]:
                    current_val = df[col].sum()

                    # Try to calculate change if possible
                    delta = None
                    delta_suffix = "vs previous period"

                    # Use the selected date column if available, otherwise detect automatically
                    if selected_date_col:
                        date_col = selected_date_col
                    else:
                        date_cols = detect_date_columns(df)
                        if date_cols:
                            date_col = date_cols[0]
                        else:
                            date_col = None

                    if date_col:
                        try:
                            # Make a copy of the dataframe to avoid modifying the original
                            df_copy = df.copy()

                            # Try to convert to datetime if it's a regular date column
                            if date_col in detect_date_columns(df):
                                try:
                                    # Ensure date column is datetime
                                    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
                                        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                                except:
                                    pass

                            # Sort the dataframe by the selected column
                            df_sorted = df_copy.sort_values(by=date_col)

                            # Different comparison logic based on selected time frame
                            if time_frame == "Half-Half":
                                # Original half-half comparison
                                half_point = len(df_sorted) // 2
                                prev_val = df_sorted.iloc[:half_point][col].sum()
                                current_val = df_sorted.iloc[half_point:][col].sum()
                                delta_suffix = "vs previous half"

                            elif time_frame == "Monthly":
                                # Add month column for grouping
                                df_sorted['month'] = df_sorted[date_col].dt.to_period('M')
                                months = df_sorted['month'].unique()

                                if len(months) >= 2:
                                    current_month = months[-1]
                                    prev_month = months[-2]

                                    current_data = df_sorted[df_sorted['month'] == current_month]
                                    prev_data = df_sorted[df_sorted['month'] == prev_month]

                                    current_val = current_data[col].sum()
                                    prev_val = prev_data[col].sum()
                                    delta_suffix = f"vs {prev_month}"
                                else:
                                    # Not enough months for comparison
                                    raise ValueError("Not enough monthly data for comparison")

                            elif time_frame == "Quarterly":
                                # Add quarter column for grouping
                                df_sorted['quarter'] = df_sorted[date_col].dt.to_period('Q')
                                quarters = df_sorted['quarter'].unique()

                                if len(quarters) >= 2:
                                    current_quarter = quarters[-1]
                                    prev_quarter = quarters[-2]

                                    current_data = df_sorted[df_sorted['quarter'] == current_quarter]
                                    prev_data = df_sorted[df_sorted['quarter'] == prev_quarter]

                                    current_val = current_data[col].sum()
                                    prev_val = prev_data[col].sum()
                                    delta_suffix = f"vs {prev_quarter}"
                                else:
                                    # Not enough quarters for comparison
                                    raise ValueError("Not enough quarterly data for comparison")

                            elif time_frame == "Yearly":
                                # Add year column for grouping
                                df_sorted['year'] = df_sorted[date_col].dt.year
                                years = df_sorted['year'].unique()

                                if len(years) >= 2:
                                    # Normal case: multiple years available
                                    current_year = years[-1]
                                    prev_year = years[-2]

                                    current_data = df_sorted[df_sorted['year'] == current_year]
                                    prev_data = df_sorted[df_sorted['year'] == prev_year]

                                    current_val = current_data[col].sum()
                                    prev_val = prev_data[col].sum()
                                    delta_suffix = f"vs {prev_year}"
                                elif len(years) == 1 and not date_col.endswith('_standardized'):
                                    # Special case: only one year available (e.g., all 2025)
                                    # Check if a standardized version of this column exists
                                    standardized_col = f"{date_col}_standardized"
                                    if standardized_col in df.columns:
                                        # Use the standardized column instead
                                        st.info(f"Using standardized date column '{standardized_col}' for yearly comparison")

                                        # Recalculate with the standardized column
                                        df_sorted_std = df.copy()
                                        df_sorted_std[standardized_col] = pd.to_datetime(df_sorted_std[standardized_col], errors='coerce')
                                        df_sorted_std = df_sorted_std.sort_values(by=standardized_col)

                                        df_sorted_std['year'] = df_sorted_std[standardized_col].dt.year
                                        years_std = df_sorted_std['year'].unique()

                                        if len(years_std) >= 2:
                                            current_year = years_std[-1]
                                            prev_year = years_std[-2]

                                            current_data = df_sorted_std[df_sorted_std['year'] == current_year]
                                            prev_data = df_sorted_std[df_sorted_std['year'] == prev_year]

                                            current_val = current_data[col].sum()
                                            prev_val = prev_data[col].sum()
                                            delta_suffix = f"vs {prev_year} (adjusted)"
                                        else:
                                            # Fall back to half-half comparison
                                            half_point = len(df_sorted) // 2
                                            prev_val = df_sorted.iloc[:half_point][col].sum()
                                            current_val = df_sorted.iloc[half_point:][col].sum()
                                            delta_suffix = "vs first half (no year variation)"
                                    else:
                                        # Fall back to half-half comparison
                                        half_point = len(df_sorted) // 2
                                        prev_val = df_sorted.iloc[:half_point][col].sum()
                                        current_val = df_sorted.iloc[half_point:][col].sum()
                                        delta_suffix = "vs first half (no year variation)"
                                else:
                                    # Not enough years for comparison
                                    raise ValueError("Not enough yearly data for comparison")

                            # Calculate percentage change
                            if prev_val != 0:
                                delta = f"{((current_val - prev_val) / prev_val * 100):.1f}%"
                            else:
                                delta = "N/A"

                        except Exception as e:
                            # If comparison fails, fall back to overall average
                            current_val = df[col].mean()
                            delta = None
                            delta_suffix = "no comparison available"

                    # Format value based on magnitude
                    if abs(current_val) >= 1e6:
                        formatted_val = f"{current_val/1e6:.2f}M"
                    elif abs(current_val) >= 1e3:
                        formatted_val = f"{current_val/1e3:.2f}K"
                    else:
                        formatted_val = f"{current_val:.2f}"

                    create_kpi_card(col, formatted_val, delta, delta_suffix)

            # Add summary statistics with improved styling
            st.write("### Summary Statistics")

            # Create a styled dataframe with better formatting and add sum
            summary_df = df[kpi_cols].describe()

            # Add sum row to the summary statistics
            sum_row = pd.DataFrame({col: [df[col].sum()] for col in kpi_cols}, index=['sum'])
            summary_df = pd.concat([sum_row, summary_df])

            # Add growth information for each column
            growth_info = {}
            for col in kpi_cols:
                try:
                    # Use the selected date column if available, otherwise detect automatically
                    if selected_date_col:
                        date_col = selected_date_col
                    else:
                        date_cols = detect_date_columns(df)
                        if date_cols:
                            date_col = date_cols[0]
                        else:
                            date_col = None

                    if date_col:
                        # Make a copy of the dataframe to avoid modifying the original
                        df_copy = df.copy()

                        # Try to convert to datetime if it's a regular date column
                        if date_col in detect_date_columns(df):
                            try:
                                # Ensure date column is datetime
                                if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
                                    df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
                            except:
                                pass

                        # Sort the dataframe by the selected column
                        df_sorted = df_copy.sort_values(by=date_col)

                        # Use the same comparison logic as for KPI cards
                        if time_frame == "Half-Half":
                            half_point = len(df_sorted) // 2
                            prev_val = df_sorted.iloc[:half_point][col].sum()
                            current_val = df_sorted.iloc[half_point:][col].sum()
                            if prev_val != 0:
                                growth = ((current_val - prev_val) / prev_val * 100)
                                growth_info[col] = f"{growth:.1f}% vs previous half"
                            else:
                                growth_info[col] = "N/A"
                        elif time_frame == "Monthly":
                            # Similar logic for monthly comparison
                            df_sorted['month'] = df_sorted[date_col].dt.to_period('M')
                            months = df_sorted['month'].unique()
                            if len(months) >= 2:
                                current_month = months[-1]
                                prev_month = months[-2]
                                current_data = df_sorted[df_sorted['month'] == current_month]
                                prev_data = df_sorted[df_sorted['month'] == prev_month]
                                current_val = current_data[col].sum()
                                prev_val = prev_data[col].sum()
                                if prev_val != 0:
                                    growth = ((current_val - prev_val) / prev_val * 100)
                                    growth_info[col] = f"{growth:.1f}% vs {prev_month}"
                                else:
                                    growth_info[col] = "N/A"
                            else:
                                growth_info[col] = "Not enough data"
                        elif time_frame == "Quarterly":
                            # Similar logic for quarterly comparison
                            df_sorted['quarter'] = df_sorted[date_col].dt.to_period('Q')
                            quarters = df_sorted['quarter'].unique()
                            if len(quarters) >= 2:
                                current_quarter = quarters[-1]
                                prev_quarter = quarters[-2]
                                current_data = df_sorted[df_sorted['quarter'] == current_quarter]
                                prev_data = df_sorted[df_sorted['quarter'] == prev_quarter]
                                current_val = current_data[col].sum()
                                prev_val = prev_data[col].sum()
                                if prev_val != 0:
                                    growth = ((current_val - prev_val) / prev_val * 100)
                                    growth_info[col] = f"{growth:.1f}% vs {prev_quarter}"
                                else:
                                    growth_info[col] = "N/A"
                            else:
                                growth_info[col] = "Not enough data"
                        elif time_frame == "Yearly":
                            # Similar logic for yearly comparison
                            df_sorted['year'] = df_sorted[date_col].dt.year
                            years = df_sorted['year'].unique()

                            if len(years) >= 2:
                                # Normal case: multiple years available
                                current_year = years[-1]
                                prev_year = years[-2]
                                current_data = df_sorted[df_sorted['year'] == current_year]
                                prev_data = df_sorted[df_sorted['year'] == prev_year]
                                current_val = current_data[col].sum()
                                prev_val = prev_data[col].sum()
                                if prev_val != 0:
                                    growth = ((current_val - prev_val) / prev_val * 100)
                                    growth_info[col] = f"{growth:.1f}% vs {prev_year}"
                                else:
                                    growth_info[col] = "N/A"
                            elif len(years) == 1 and not date_col.endswith('_standardized'):
                                # Special case: only one year available (e.g., all 2025)
                                # Check if a standardized version of this column exists
                                standardized_col = f"{date_col}_standardized"
                                if standardized_col in df.columns:
                                    # Use the standardized column instead
                                    # Recalculate with the standardized column
                                    df_sorted_std = df.copy()
                                    df_sorted_std[standardized_col] = pd.to_datetime(df_sorted_std[standardized_col], errors='coerce')
                                    df_sorted_std = df_sorted_std.sort_values(by=standardized_col)

                                    df_sorted_std['year'] = df_sorted_std[standardized_col].dt.year
                                    years_std = df_sorted_std['year'].unique()

                                    if len(years_std) >= 2:
                                        current_year = years_std[-1]
                                        prev_year = years_std[-2]

                                        current_data = df_sorted_std[df_sorted_std['year'] == current_year]
                                        prev_data = df_sorted_std[df_sorted_std['year'] == prev_year]

                                        current_val = current_data[col].sum()
                                        prev_val = prev_data[col].sum()

                                        if prev_val != 0:
                                            growth = ((current_val - prev_val) / prev_val * 100)
                                            growth_info[col] = f"{growth:.1f}% vs {prev_year} (adjusted)"
                                        else:
                                            growth_info[col] = "N/A"
                                    else:
                                        # Fall back to half-half comparison
                                        half_point = len(df_sorted) // 2
                                        prev_val = df_sorted.iloc[:half_point][col].sum()
                                        current_val = df_sorted.iloc[half_point:][col].sum()

                                        if prev_val != 0:
                                            growth = ((current_val - prev_val) / prev_val * 100)
                                            growth_info[col] = f"{growth:.1f}% vs first half (no year variation)"
                                        else:
                                            growth_info[col] = "N/A"
                                else:
                                    # Fall back to half-half comparison
                                    half_point = len(df_sorted) // 2
                                    prev_val = df_sorted.iloc[:half_point][col].sum()
                                    current_val = df_sorted.iloc[half_point:][col].sum()

                                    if prev_val != 0:
                                        growth = ((current_val - prev_val) / prev_val * 100)
                                        growth_info[col] = f"{growth:.1f}% vs first half (no year variation)"
                                    else:
                                        growth_info[col] = "N/A"
                            else:
                                growth_info[col] = "Not enough data"
                    else:
                        growth_info[col] = "No date column"
                except Exception:
                    growth_info[col] = "Calculation error"

            # Display the summary statistics
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=False,
                column_config={
                    col: st.column_config.NumberColumn(
                        col,
                        format="%.2f",
                        help=growth_info.get(col, "")
                    ) for col in kpi_cols
                }
            )

            # Display growth information below the table
            st.write("#### Growth Information")
            for col, info in growth_info.items():
                st.write(f"**{col}**: {info}")

            # Add a note about the comparison period
            st.info(f"📊 KPI metrics are currently showing comparison based on **{time_frame}** periods. You can change this using the selector above.")
        else:
            st.info("Please select at least one column for KPIs.")
    else:
        st.info("No numeric columns found for KPIs.")
