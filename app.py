"""
Main entry point for the Financial Data Visualization application.
"""
import streamlit as st
from streamlit_extras.colored_header import colored_header

# Import configuration
from config import APP_TITLE, APP_ICON, APP_LAYOUT, INITIAL_SIDEBAR_STATE

# Version indicator to verify deployment
APP_VERSION = "1.2.0-sidebyside"  # Update this when making significant changes

# Set a flag for side-by-side charts (we'll check the actual module later)
# This avoids import issues that could affect st.set_page_config()
SIDE_BY_SIDE_CHARTS_ENABLED = True

# Import components
from components.theme import initialize_theme
from components.sidebar import create_sidebar

# Import utilities
from utils.data_utils import load_excel_file, get_column_types
import traceback

# Import visualization modules
from visualizations.basic_charts import show_data_preview, show_data_statistics
from visualizations.advanced_charts import show_advanced_visualizations
from visualizations.smart_visuals import show_smart_visualizations

# Import analytics modules
from analytics.statistical import show_advanced_analytics
from analytics.growth_analysis import show_growth_analysis

# Import forecasting modules
from forecasting.forecasting import show_forecasting

# Import PDF report module
from reports.pdf_report import show_pdf_reports

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Function to clear Streamlit cache
def clear_cache():
    """Clear all Streamlit caches to ensure a fresh start."""
    # Clear all cached functions
    try:
        # Try to clear all caches
        for attr in dir(st):
            if attr.startswith('cache_') and callable(getattr(st, f"{attr}_clear")):
                getattr(st, f"{attr}_clear")()
    except Exception:
        # If any error occurs, try individual cache clearing methods
        try:
            if hasattr(st, 'cache_data_clear'):
                st.cache_data_clear()
        except:
            pass

        try:
            if hasattr(st, 'cache_resource_clear'):
                st.cache_resource_clear()
        except:
            pass

# Function to display user-friendly error messages
def handle_error(error, error_location="application"):
    """Display a user-friendly error message with details for debugging."""
    error_details = traceback.format_exc()

    st.error(f"⚠️ Error in {error_location}: {str(error)}")

    with st.expander("Error Details (for developers)", expanded=False):
        st.code(error_details)

    st.markdown("""
    ### Troubleshooting Tips:
    - Try refreshing the page
    - Clear your browser cache
    - Try a different browser
    - If the issue persists, please contact support
    """)

    return False

# Initialize theme
try:
    initialize_theme()
except Exception as e:
    handle_error(e, "theme initialization")

# App title and description
try:
    colored_header(
        label=APP_TITLE,
        description="Upload an Excel file to visualize and analyze your financial data",
        color_name="blue-70"
    )
except Exception as e:
    st.title(APP_TITLE)
    st.write("Upload an Excel file to visualize and analyze your financial data")
    handle_error(e, "header rendering")

# Create sidebar navigation
selected_section = create_sidebar()

# Initialize session state for data persistence
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.numeric_cols = []
    st.session_state.categorical_cols = []
    st.session_state.file_name = None
    st.session_state.last_modified = None

# Initialize process control flags in session state if not present
if 'active_processes' not in st.session_state:
    st.session_state.active_processes = {}

# Check for URL query parameters that might indicate a reset request
# This helps ensure the page is truly reset after the Close File Execution button is clicked
if st.query_params.get('reset', ['false'])[0] == 'true':
    # Clear all Streamlit caches
    clear_cache()

    # Clear all session state except theme preferences
    dark_mode = st.session_state.get('dark_mode', False)

    # Get all keys to remove
    keys_to_remove = list(st.session_state.keys())

    # Remove all session state variables
    for key in keys_to_remove:
        if key != 'dark_mode':  # Preserve theme preference
            try:
                del st.session_state[key]
            except:
                pass

    # Restore theme preference
    st.session_state.dark_mode = dark_mode

    # Reinitialize essential session state variables
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.numeric_cols = []
    st.session_state.categorical_cols = []
    st.session_state.file_name = None
    st.session_state.last_modified = None
    st.session_state.active_processes = {}

    # Clear the query parameter
    st.query_params.clear()

    # Show a success message
    st.success("Session reset complete. You can upload a new file.")

# File uploader section with title
st.subheader("📂 Data Upload")
st.markdown("Upload your Excel file to begin analysis.")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# No stop button needed

# Load data when file is uploaded or changed
if uploaded_file is not None:
    try:
        # Check if we need to reload the data
        file_details_changed = (
            not st.session_state.data_loaded or
            st.session_state.file_name != uploaded_file.name or
            (hasattr(uploaded_file, 'last_modified') and
             st.session_state.last_modified != uploaded_file.last_modified)
        )

        if file_details_changed:
            with st.spinner('Loading data... This may take a moment for large files.'):
                try:
                    # Load the data
                    df = load_excel_file(uploaded_file)

                    if df is not None:
                        # Store dataframe and metadata in session state
                        st.session_state.df = df
                        st.session_state.file_name = uploaded_file.name
                        if hasattr(uploaded_file, 'last_modified'):
                            st.session_state.last_modified = uploaded_file.last_modified

                        # Get column types for visualization (cached function)
                        st.session_state.numeric_cols, st.session_state.categorical_cols = get_column_types(df)
                        st.session_state.data_loaded = True

                        # Show success message
                        st.success(f"Successfully loaded {uploaded_file.name} with {len(df)} rows and {len(df.columns)} columns")
                except Exception as e:
                    handle_error(e, "data loading")
    except Exception as e:
        handle_error(e, "file processing")

    # Use the data from session state
    if st.session_state.data_loaded:
        df = st.session_state.df
        numeric_cols = st.session_state.numeric_cols
        categorical_cols = st.session_state.categorical_cols

        # Display content based on selected section
        if selected_section == "Data Preview":
            show_data_preview(df)

        elif selected_section == "Statistical Details":
            show_data_statistics(df, numeric_cols)

        elif selected_section == "Data Visualization":
            # Display deployment verification message
            if SIDE_BY_SIDE_CHARTS_ENABLED:
                st.success("✅ Side-by-side charts feature is enabled in this deployment.")
            else:
                st.warning("⚠️ Side-by-side charts feature may not be available in this deployment. Please check for updates.")

            # Add a performance note
            with st.expander("Performance Tips", expanded=False):
                st.markdown("""
                **Performance Tips:**
                - Use data aggregation (day, week, month) for large time series
                - Disable rolling averages if not needed
                - For very large datasets, consider filtering by date range
                - Charts are cached for better performance when revisiting
                """)

            show_advanced_visualizations(df, numeric_cols, categorical_cols)

        elif selected_section == "Smart Visuals":
            # Add an info note about smart visualization
            with st.expander("About Smart Visuals", expanded=True):
                st.markdown("""
                **Smart Visuals** automatically analyzes your data to suggest the most appropriate
                visualizations. Simply select a chart type, and the system will:

                - Identify relevant columns for that chart type
                - Configure appropriate options and filters
                - Generate an optimized visualization
                - Provide insights about what the chart shows

                You can always adjust the suggestions to customize your visualization.
                """)

            show_smart_visualizations(df, numeric_cols, categorical_cols)

        elif selected_section == "Advanced Analytics":
            show_advanced_analytics(df, numeric_cols, categorical_cols)

        elif selected_section == "Growth Trends":
            # Add an info note about growth analysis
            with st.expander("About Growth Trends Analysis", expanded=True):
                st.markdown("""
                **Growth Trends Analysis** helps you understand year-over-year changes in your data:

                - Calculate yearly growth (delta) and growth percentage
                - Detect trends and unusual patterns
                - Compare actual values against claimed forecasts
                - Generate simple predictions for future periods

                This analysis is particularly useful for financial metrics and performance indicators.
                """)

            show_growth_analysis(df, numeric_cols)

        elif selected_section == "Forecasting":
            show_forecasting(df, numeric_cols)

        elif selected_section == "PDF Reports":
            show_pdf_reports(df, st.session_state.file_name, numeric_cols, categorical_cols)
        elif selected_section == "Agentic Analysis":
            st.markdown(
                """
                <div style="text-align:center; margin-top:2em;">
                    <a href="http://localhost:5173/" target="_blank" style="font-size:1.2em; font-weight:bold; color:#2563eb;">
                        🚀 Open Agentic Analysis (new tab)
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Please upload an Excel file to begin.")
