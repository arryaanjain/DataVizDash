"""
Basic visualization components for the application.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import VIZ_TABS

def show_data_preview(df):
    """Show a preview of the data with enhanced insights."""
    st.subheader("Data Preview")

    # Create an expandable section for the data preview
    with st.expander("View Data Sample", expanded=True):
        st.dataframe(df.head(10))

    # Dataset Structure Overview
    st.subheader("Dataset Structure")

    # Basic dataset dimensions
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
    with col2:
        numeric_count = len(df.select_dtypes(include=['float64', 'int64']).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        st.metric("Numeric Columns", numeric_count)
        st.metric("Categorical Columns", categorical_count)

    # Data Quality Overview
    st.subheader("Data Quality Overview")

    # Calculate null values
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

    # Display null values information
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Missing Cells", missing_cells)
    with col2:
        st.metric("Missing Percentage", f"{missing_percentage:.2f}%")
    with col3:
        st.metric("Complete Cells", total_cells - missing_cells)

    # Show columns with missing values if any exist
    if missing_cells > 0:
        with st.expander("Columns with Missing Values"):
            missing_by_column = df.isna().sum()
            missing_by_column = missing_by_column[missing_by_column > 0].sort_values(ascending=False)

            if not missing_by_column.empty:
                missing_df = pd.DataFrame({
                    'Column': missing_by_column.index,
                    'Missing Values': missing_by_column.values,
                    'Percentage': (missing_by_column.values / df.shape[0] * 100).round(2)
                })
                st.dataframe(missing_df)
            else:
                st.info("No columns with missing values.")

    # Data Range Overview for numeric columns
    if numeric_count > 0:
        st.subheader("Numeric Data Range")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

        # Create a dataframe with min, max, mean for each numeric column
        range_data = []
        for col in numeric_cols:
            if not df[col].isna().all():  # Skip columns with all NaN values
                range_data.append({
                    'Column': col,
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Mean': df[col].mean(),
                    'Median': df[col].median(),
                    'Std Dev': df[col].std()
                })

        if range_data:
            range_df = pd.DataFrame(range_data)
            st.dataframe(range_df)

    # Categorical Data Overview
    if categorical_count > 0:
        st.subheader("Categorical Data Overview")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Create expandable sections for each categorical column
        for col in categorical_cols:
            if not df[col].isna().all():  # Skip columns with all NaN values
                with st.expander(f"{col} - Unique Values: {df[col].nunique()}"):
                    # Get value counts and calculate percentages
                    value_counts = df[col].value_counts().reset_index()
                    value_counts.columns = ['Value', 'Count']
                    value_counts['Percentage'] = (value_counts['Count'] / df.shape[0] * 100).round(2)

                    # Display only top 10 values if there are more than 10 unique values
                    if len(value_counts) > 10:
                        st.write(f"Showing top 10 out of {len(value_counts)} unique values")
                        st.dataframe(value_counts.head(10))
                    else:
                        st.dataframe(value_counts)

def show_data_statistics(df, numeric_cols):
    """Show data statistics and correlation heatmap."""
    st.subheader("Data Statistics")
    st.write(df.describe())

    # Show correlation heatmap if there are not too many columns (to avoid clutter)
    if 1 < len(numeric_cols) <= 10:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
        plt.title('Correlation Heatmap')
        st.pyplot(fig)

def show_numeric_visualization(df, numeric_cols):
    """Show visualizations for numeric data."""
    if numeric_cols:
        st.write("### Numeric Data Visualization")

        # Histogram
        selected_num_col = st.selectbox("Select a numeric column for histogram", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_num_col].dropna(), kde=True, ax=ax)
        plt.title(f'Histogram of {selected_num_col}')
        st.pyplot(fig)
    else:
        st.info("No numeric columns found in the data.")

def show_categorical_visualization(df, categorical_cols):
    """Show visualizations for categorical data."""
    if categorical_cols:
        st.write("### Categorical Data Visualization")

        selected_cat_col = st.selectbox("Select a categorical column for count plot", categorical_cols)
        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts = df[selected_cat_col].value_counts().sort_values(ascending=False).head(10)
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
        plt.title(f'Count Plot of {selected_cat_col} (Top 10)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No categorical columns found in the data.")

def show_relationship_visualization(df, numeric_cols):
    """Show visualizations for relationships between variables."""
    if len(numeric_cols) >= 2:
        st.write("### Scatter Plot")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis column", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        st.pyplot(fig)
    else:
        st.info("At least two numeric columns are required for relationship plots.")

def show_basic_visualizations(df, numeric_cols, categorical_cols):
    """Show basic visualizations using tabs."""
    st.subheader("Data Visualization")

    # Create tabs for different visualization types
    viz_tabs = st.tabs(VIZ_TABS[:3])  # Use only the first 3 tabs for basic visualizations

    # Numeric data visualization tab
    with viz_tabs[0]:
        show_numeric_visualization(df, numeric_cols)

    # Categorical data visualization tab
    with viz_tabs[1]:
        show_categorical_visualization(df, categorical_cols)

    # Relationships visualization tab
    with viz_tabs[2]:
        show_relationship_visualization(df, numeric_cols)
