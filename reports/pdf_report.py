"""
PDF report generation module for the application.
This module provides functionality to generate comprehensive PDF reports
from the uploaded dataset, including metadata, statistics, and visualizations.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import traceback
from fpdf import FPDF
from datetime import datetime

# Import utilities
from utils.data_utils import detect_date_columns

# Define constants for PDF generation
PAGE_WIDTH = 210  # A4 width in mm
PAGE_HEIGHT = 297  # A4 height in mm
MARGIN = 15  # Margin in mm (increased for better text rendering)
CONTENT_WIDTH = PAGE_WIDTH - (2 * MARGIN)

class DataReportPDF(FPDF):
    """Custom PDF class for data reports with header and footer."""

    def __init__(self, title="Data Analysis Report"):
        super().__init__()
        self.title = title
        # Increase margin for better text rendering
        self.set_auto_page_break(auto=True, margin=20)
        # Set default margins
        self.set_margins(left=MARGIN, top=MARGIN, right=MARGIN)

    def header(self):
        """Add header to each page."""
        # Set font for header
        self.set_font('Arial', 'B', 12)

        # Add logo if needed (commented out for now)
        # self.image('logo.png', 10, 8, 33)

        # Add title
        self.cell(0, 10, self.title, 0, 1, 'C')

        # Add date
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')

        # Add line
        self.line(MARGIN, 25, PAGE_WIDTH - MARGIN, 25)

        # Add space after header
        self.ln(5)

    def footer(self):
        """Add footer to each page."""
        # Set position at 2 cm from bottom (increased for better spacing)
        self.set_y(-20)

        # Set font for footer
        self.set_font('Arial', 'I', 8)

        # Add page number
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(df, file_name, numeric_cols, categorical_cols, options=None):
    """
    Generate a comprehensive PDF report from the dataset.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    file_name : str
        The name of the uploaded file
    numeric_cols : list
        List of numeric column names
    categorical_cols : list
        List of categorical column names
    options : dict, optional
        Dictionary of report options (which sections to include)

    Returns:
    --------
    bytes
        The PDF report as bytes for download
    """
    # Default options if none provided
    if options is None:
        options = {
            'include_dataset_overview': True,
            'include_summary_stats': True,
            'include_visualizations': True,
            'include_comparative': True
        }

    # Create PDF object
    pdf = DataReportPDF(title=f"Data Analysis Report: {file_name}")
    pdf.add_page()

    # Add dataset overview
    if options.get('include_dataset_overview', True):
        add_dataset_overview(pdf, df, file_name, numeric_cols, categorical_cols)

    # Add summary statistics for numeric columns
    if options.get('include_summary_stats', True) and numeric_cols:
        add_summary_statistics(pdf, df, numeric_cols)

    # Add visualizations (pie charts and stacked bar charts)
    if options.get('include_visualizations', True) and categorical_cols:
        add_visualizations(pdf, df, numeric_cols, categorical_cols)

    # Add comparative analysis if there are multiple numeric columns
    if options.get('include_comparative', True) and len(numeric_cols) > 1:
        add_comparative_analysis(pdf, df, numeric_cols, categorical_cols)

    # Return PDF as bytes
    # In newer versions of fpdf2, output() already returns bytes
    pdf_output = pdf.output(dest='S')
    # Check if encoding is needed (for compatibility with different fpdf2 versions)
    if isinstance(pdf_output, str):
        return pdf_output.encode('latin1')
    else:
        return pdf_output

def add_dataset_overview(pdf, df, file_name, numeric_cols, categorical_cols):
    """Add dataset overview to the PDF."""
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "1. Dataset Overview", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, f"This report provides an analysis of the dataset '{file_name}'.")
    pdf.ln(5)

    # Dataset dimensions
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Dataset Dimensions", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.cell(60, 7, "Total Rows:", 0, 0)
    pdf.cell(0, 7, f"{df.shape[0]}", 0, 1)

    pdf.cell(60, 7, "Total Columns:", 0, 0)
    pdf.cell(0, 7, f"{df.shape[1]}", 0, 1)

    # Data quality overview
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0

    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Data Quality Overview", 0, 1)

    # Create a table for missing values per column
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 7, "Column", 1, 0, 'C')
    pdf.cell(40, 7, "Missing Values", 1, 0, 'C')
    pdf.cell(40, 7, "Missing %", 1, 1, 'C')

    pdf.set_font('Arial', '', 10)
    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / len(df)) * 100

        pdf.cell(60, 7, str(col), 1, 0)
        pdf.cell(40, 7, str(missing), 1, 0, 'R')
        pdf.cell(40, 7, f"{missing_pct:.2f}%", 1, 1, 'R')

    # Overall data completeness
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 7, "Overall Data Completeness:", 0, 0)
    pdf.cell(0, 7, f"{100 - missing_percentage:.2f}%", 0, 1)

    # Numeric data range
    if numeric_cols:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Numeric Data Range", 0, 1)

        # Create a table for numeric data ranges
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(40, 7, "Column", 1, 0, 'C')
        pdf.cell(30, 7, "Min", 1, 0, 'C')
        pdf.cell(30, 7, "Max", 1, 0, 'C')
        pdf.cell(30, 7, "Mean", 1, 0, 'C')
        pdf.cell(30, 7, "Median", 1, 1, 'C')

        pdf.set_font('Arial', '', 10)
        for col in numeric_cols:
            if not df[col].isna().all():  # Skip columns with all NaN values
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                median_val = df[col].median()

                pdf.cell(40, 7, str(col), 1, 0)
                pdf.cell(30, 7, f"{min_val:.2f}", 1, 0, 'R')
                pdf.cell(30, 7, f"{max_val:.2f}", 1, 0, 'R')
                pdf.cell(30, 7, f"{mean_val:.2f}", 1, 0, 'R')
                pdf.cell(30, 7, f"{median_val:.2f}", 1, 1, 'R')

    # Categorical data overview (only if unique values < 10)
    if categorical_cols:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Categorical Data Overview", 0, 1)

        pdf.set_font('Arial', '', 10)
        for col in categorical_cols:
            if not df[col].isna().all():  # Skip columns with all NaN values
                unique_values = df[col].value_counts()

                # Only show if there are fewer than 10 unique values
                if len(unique_values) < 10:
                    pdf.set_font('Arial', 'B', 10)
                    pdf.cell(0, 7, f"{col}:", 0, 1)

                    pdf.set_font('Arial', '', 10)
                    for value, count in unique_values.items():
                        percentage = (count / len(df)) * 100
                        # Convert value to string and replace any non-Latin1 characters
                        safe_value = str(value)
                        # Replace common currency symbols with their text equivalents
                        safe_value = safe_value.replace('₹', '[Rupee]')
                        safe_value = safe_value.replace('€', '[Euro]')
                        safe_value = safe_value.replace('£', '[Pound]')
                        safe_value = safe_value.replace('¥', '[Yen]')
                        # Replace any other non-Latin1 characters with '?'
                        safe_value = ''.join(c if ord(c) < 256 else '?' for c in safe_value)

                        pdf.cell(10, 7, "", 0, 0)  # Indent
                        pdf.cell(0, 7, f"{safe_value}: {count} ({percentage:.2f}%)", 0, 1)

    pdf.ln(5)

def add_summary_statistics(pdf, df, numeric_cols):
    """Add summary statistics for numeric columns to the PDF."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "2. Summary Statistics", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "This section provides standard descriptive statistics for all numeric columns in the dataset.")
    pdf.ln(5)

    # Skip if no numeric columns
    if not numeric_cols:
        pdf.set_font('Arial', 'I', 10)
        # Use cell instead of multi_cell for better rendering
        pdf.cell(0, 5, "No numeric columns available for statistics", 0, 1)
        return

    # Get summary statistics
    stats_df = df[numeric_cols].describe().T

    # Create a table for the statistics
    pdf.set_font('Arial', 'B', 10)

    # Calculate column widths based on available space and number of columns
    stats_cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    col_width = min(25, CONTENT_WIDTH / (len(stats_cols) + 1))

    # Table header
    pdf.cell(40, 7, "Column", 1, 0, 'C')
    for col in stats_cols:
        if col in stats_df.columns:
            pdf.cell(col_width, 7, str(col), 1, 0, 'C')
    pdf.ln()

    # Table data
    pdf.set_font('Arial', '', 8)
    for idx, row in stats_df.iterrows():
        # Truncate column name if too long
        col_name = str(idx)
        if len(col_name) > 20:
            col_name = col_name[:17] + "..."

        pdf.cell(40, 7, col_name, 1, 0, 'L')

        for col in stats_cols:
            if col in stats_df.columns:
                value = row[col]
                # Format the value based on its magnitude
                try:
                    if pd.isna(value):
                        formatted_value = "N/A"
                    elif abs(float(value)) < 0.01 or abs(float(value)) > 1000:
                        formatted_value = f"{value:.2e}"
                    else:
                        formatted_value = f"{value:.2f}"
                except (ValueError, TypeError):
                    # Handle non-numeric values
                    formatted_value = str(value)

                pdf.cell(col_width, 7, formatted_value, 1, 0, 'R')
        pdf.ln()

    pdf.ln(5)

def add_visualizations(pdf, df, numeric_cols, categorical_cols):
    """Add visualizations (pie charts and stacked bar charts) to the PDF."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "3. Data Visualizations", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "This section provides visualizations for the categorical data in the dataset.")
    pdf.ln(5)

    # Check if there are any categorical columns to visualize
    if not categorical_cols:
        pdf.set_font('Arial', 'I', 10)
        # Use cell instead of multi_cell for better rendering
        pdf.cell(0, 5, "No categorical columns available for visualization", 0, 1)
        return

    # Create pie charts for categorical columns with < 10 unique values
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Pie Charts", 0, 1)

    pie_charts_created = 0

    for i, col in enumerate(categorical_cols):
        try:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue

            # Get value counts
            value_counts = df[col].value_counts()

            # Skip if there are no values or too many unique values
            if len(value_counts) == 0 or len(value_counts) >= 10:
                continue

            # Create pie chart
            plt.figure(figsize=(7, 5))
            plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%',
                   shadow=False, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(f"Distribution of {col}")
            plt.tight_layout()

            # Save figure to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                plt.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                plt.close()

                # Add the image to the PDF
                pdf.set_font('Arial', 'B', 11)
                # Use safe version of column name
                safe_col = ''.join(c if ord(c) < 256 else '?' for c in col)
                pdf.cell(0, 10, f"Distribution of {safe_col}", 0, 1)
                pdf.image(tmpfile.name, x=MARGIN, w=CONTENT_WIDTH)
                pdf.ln(5)

                # Remove the temporary file
                os.unlink(tmpfile.name)

                pie_charts_created += 1

            # Add a new page after every 2 visualizations (except the last one)
            if pie_charts_created % 2 == 0 and i < len(categorical_cols) - 1:
                pdf.add_page()

        except Exception as e:
            # Log the error and continue with the next column
            pdf.set_font('Arial', 'I', 10)
            # Use cell instead of multi_cell for better rendering
            pdf.cell(0, 5, f"Error generating pie chart for {col}", 0, 1)
            plt.close()  # Make sure to close any open figures

    if pie_charts_created == 0:
        pdf.set_font('Arial', 'I', 10)
        # Use a shorter message to avoid text rendering issues
        pdf.cell(0, 5, "No suitable categorical columns for pie charts (all have 10+ unique values)", 0, 1)

    # Create stacked bar charts if we have both categorical and numeric columns
    if categorical_cols and numeric_cols:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Stacked Bar Charts", 0, 1)

        stacked_charts_created = 0

        # Find categorical columns with reasonable number of categories
        suitable_cat_cols = []
        for col in categorical_cols:
            if not df[col].isna().all() and len(df[col].value_counts()) < 10:
                suitable_cat_cols.append(col)

        # Find numeric columns for values
        suitable_num_cols = []
        for col in numeric_cols:
            if not df[col].isna().all():
                suitable_num_cols.append(col)

        # Limit to first 3 of each to avoid too many charts
        suitable_cat_cols = suitable_cat_cols[:3]
        suitable_num_cols = suitable_num_cols[:3]

        # Create stacked bar charts
        for cat_col in suitable_cat_cols:
            for num_col in suitable_num_cols:
                try:
                    # Create a pivot table for the stacked bar chart
                    pivot_data = df.pivot_table(
                        values=num_col,
                        index=cat_col,
                        aggfunc='mean'
                    ).sort_values(by=num_col, ascending=False)

                    # Skip if pivot table is empty
                    if len(pivot_data) == 0:
                        continue

                    # Create stacked bar chart
                    plt.figure(figsize=(8, 5))
                    pivot_data.plot(kind='bar', stacked=False)

                    # Use safe versions of column names for plot labels
                    safe_num_col = ''.join(c if ord(c) < 256 else '?' for c in num_col)
                    safe_cat_col = ''.join(c if ord(c) < 256 else '?' for c in cat_col)

                    plt.title(f"Average {safe_num_col} by {safe_cat_col}")
                    plt.xlabel(safe_cat_col)
                    plt.ylabel(f"Average {safe_num_col}")
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()

                    # Save figure to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                        plt.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                        plt.close()

                        # Add the image to the PDF
                        pdf.set_font('Arial', 'B', 11)
                        # Use safe versions of column names
                        safe_num_col = ''.join(c if ord(c) < 256 else '?' for c in num_col)
                        safe_cat_col = ''.join(c if ord(c) < 256 else '?' for c in cat_col)
                        pdf.cell(0, 10, f"Average {safe_num_col} by {safe_cat_col}", 0, 1)
                        pdf.image(tmpfile.name, x=MARGIN, w=CONTENT_WIDTH)
                        pdf.ln(5)

                        # Remove the temporary file
                        os.unlink(tmpfile.name)

                        stacked_charts_created += 1

                    # Add a new page after each chart (except the last one)
                    if stacked_charts_created < len(suitable_cat_cols) * len(suitable_num_cols):
                        pdf.add_page()

                except Exception as e:
                    # Log the error and continue with the next column
                    pdf.set_font('Arial', 'I', 10)
                    # Use cell instead of multi_cell for better rendering
                    pdf.cell(0, 5, f"Error generating stacked bar chart for {cat_col}/{num_col}", 0, 1)
                    plt.close()  # Make sure to close any open figures

        if stacked_charts_created == 0:
            pdf.set_font('Arial', 'I', 10)
            # Use cell instead of multi_cell for better rendering
            pdf.cell(0, 5, "No suitable column combinations found for stacked bar charts", 0, 1)

    pdf.ln(5)

def add_comparative_analysis(pdf, df, numeric_cols, _):
    """Add comparative analysis to the PDF.

    Note: The categorical_cols parameter is not used but kept for API consistency.
    """
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "4. Comparative Analysis", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "This section compares key variables and their relationships in the dataset.")
    pdf.ln(5)

    # Check if we have enough numeric columns for correlation
    if len(numeric_cols) < 2:
        pdf.set_font('Arial', 'I', 10)
        # Use cell instead of multi_cell for better rendering
        pdf.cell(0, 5, "At least two numeric columns are required for comparative analysis", 0, 1)
        return

    try:
        # Create correlation matrix (minimal version)
        plt.figure(figsize=(10, 8))

        # Filter out columns with all NaN values
        valid_cols = [col for col in numeric_cols if not df[col].isna().all()]

        # Check if we still have enough columns
        if len(valid_cols) < 2:
            pdf.set_font('Arial', 'I', 10)
            # Use cell instead of multi_cell for better rendering
            pdf.cell(0, 5, "Not enough valid numeric columns for comparative analysis", 0, 1)
            plt.close()
            return

        # Limit to top 8 columns to keep the matrix readable
        if len(valid_cols) > 8:
            valid_cols = valid_cols[:8]

        correlation = df[valid_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()

        # Save figure to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            plt.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
            plt.close()

            # Add the image to the PDF
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, "Correlation Between Numeric Variables", 0, 1)
            pdf.image(tmpfile.name, x=MARGIN, w=CONTENT_WIDTH)

            # Remove the temporary file
            os.unlink(tmpfile.name)

        pdf.ln(5)

        # Add a brief explanation of correlation
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, "The correlation matrix shows the strength of relationships between numeric variables. Values close to 1 indicate strong positive correlation, values close to -1 indicate strong negative correlation, and values close to 0 indicate little to no correlation.")

        # Create a scatter plot for the two most correlated variables
        if len(valid_cols) >= 2:
            # Find the two most correlated variables (excluding self-correlations)
            corr_pairs = []
            for i in range(len(valid_cols)):
                for j in range(i+1, len(valid_cols)):
                    corr_value = abs(correlation.iloc[i, j])
                    corr_pairs.append((valid_cols[i], valid_cols[j], corr_value))

            # Sort by correlation value (descending)
            corr_pairs.sort(key=lambda x: x[2], reverse=True)

            if corr_pairs:
                # Get the most correlated pair
                var1, var2, corr_val = corr_pairs[0]

                # Create scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(df[var1], df[var2], alpha=0.5)

                # Use safe versions of variable names for plot labels
                safe_var1 = ''.join(c if ord(c) < 256 else '?' for c in var1)
                safe_var2 = ''.join(c if ord(c) < 256 else '?' for c in var2)

                plt.title(f"Relationship between {safe_var1} and {safe_var2}\nCorrelation: {corr_val:.2f}")
                plt.xlabel(safe_var1)
                plt.ylabel(safe_var2)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save figure to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                    plt.savefig(tmpfile.name, dpi=300, bbox_inches='tight')
                    plt.close()

                    # Add the image to the PDF
                    pdf.ln(10)
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Scatter Plot of Most Correlated Variables", 0, 1)
                    pdf.image(tmpfile.name, x=MARGIN, w=CONTENT_WIDTH)

                    # Remove the temporary file
                    os.unlink(tmpfile.name)

    except Exception as e:
        # Log the error
        pdf.set_font('Arial', 'I', 10)
        # Use cell instead of multi_cell for better rendering
        pdf.cell(0, 5, "Error generating comparative analysis", 0, 1)
        plt.close()  # Make sure to close any open figures

    pdf.ln(5)

def add_conclusion(pdf, df):
    """Add conclusion to the PDF."""
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, "6. Conclusion", 0, 1)

    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 5, "This report provides a comprehensive overview of the dataset, including metadata, summary statistics, and visualizations. The insights derived from this analysis can be used to better understand the data and make informed decisions.")
    pdf.ln(5)

    # Add date columns if available
    date_cols = detect_date_columns(df)
    if date_cols:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Time-Based Analysis Potential:", 0, 1)

        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 5, f"The dataset contains {len(date_cols)} date/time columns: {', '.join(date_cols)}. These columns can be used for time series analysis, trend detection, and forecasting.")
        pdf.ln(5)

    # Add final notes
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Next Steps:", 0, 1)

    pdf.set_font('Arial', '', 10)
    # Use standard hyphens instead of bullet points to avoid Unicode issues
    pdf.multi_cell(0, 5, "- Explore relationships between specific variables of interest\n- Conduct more detailed statistical analysis\n- Consider creating predictive models based on the data\n- Investigate any anomalies or outliers identified in the visualizations")

def show_pdf_reports(df, file_name, numeric_cols, categorical_cols):
    """Show PDF report generation UI and handle the generation process."""
    st.subheader("PDF Report Generation")

    st.markdown("""
    Generate a comprehensive PDF report that summarizes your dataset's key information,
    statistics, and visualizations. This feature is designed for non-technical users who
    want a quick and insightful overview of their data.
    """)

    # Create expandable section for report options
    with st.expander("Report Options", expanded=True):
        st.markdown("### Report Contents")

        # Checkboxes for report sections (all selected by default)
        include_dataset_overview = st.checkbox("Dataset Overview", value=True,
                                      help="Include dataset dimensions, data quality, and column information")

        include_summary_stats = st.checkbox("Summary Statistics", value=True,
                                   help="Include standard descriptive statistics for numeric columns")

        include_visualizations = st.checkbox("Data Visualizations", value=True,
                                     help="Include pie charts and stacked bar charts")

        include_comparative = st.checkbox("Comparative Analysis", value=True,
                                  help="Include correlation analysis and key variable comparisons")

        # Collect options into a dictionary
        report_options = {
            'include_dataset_overview': include_dataset_overview,
            'include_summary_stats': include_summary_stats,
            'include_visualizations': include_visualizations,
            'include_comparative': include_comparative
        }

    # Generate report button
    if st.button("📄 Generate PDF Report"):
        with st.spinner("Generating PDF report... This may take a moment."):
            try:
                # Generate the PDF report with selected options
                pdf_bytes = generate_pdf_report(df, file_name, numeric_cols, categorical_cols, report_options)

                # Create download button for the PDF
                pdf_filename = f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

                # Convert bytearray to bytes if needed
                if isinstance(pdf_bytes, bytearray):
                    pdf_bytes = bytes(pdf_bytes)

                # Use Streamlit's download_button for better integration
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    help="Click to download the generated PDF report"
                )

                # Show success message
                st.success("PDF report generated successfully! Click the button above to download.")

                # Add a preview note
                st.info("The PDF report includes key information about your data, formatted for easy sharing and presentation.")

            except Exception as e:
                st.error(f"Error generating PDF report: {str(e)}")
                st.info("Try again with a smaller dataset or fewer visualizations if the error persists.")

                # Show detailed error information for debugging
                with st.expander("Error details", expanded=False):
                    st.code(traceback.format_exc())
