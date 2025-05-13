"""
Sidebar navigation components for the application.
"""
import streamlit as st
from streamlit_option_menu import option_menu
from config import NAV_OPTIONS, NAV_ICONS
import sys

def create_sidebar():
    """Create the sidebar navigation with data management options."""
    with st.sidebar:
        st.header("Navigation")

        # Create sidebar navigation
        selected_section = option_menu(
            menu_title=None,
            options=NAV_OPTIONS,
            icons=NAV_ICONS,
            menu_icon="cast",
            default_index=0,
        )

        # Store the selected section in session state
        if 'selected_section' not in st.session_state:
            st.session_state.selected_section = selected_section
        else:
            st.session_state.selected_section = selected_section

        # Add a separator
        st.markdown("---")

        # Add Data Management section
        data_mgmt_col1, data_mgmt_col2 = st.columns([3, 1])
        with data_mgmt_col1:
            st.markdown("### Data Management")
        with data_mgmt_col2:
            # Show status indicator
            if st.session_state.get('data_loaded', False):
                st.markdown("🟢")  # Green circle for data loaded
            else:
                st.markdown("⚪")  # White circle for no data

        # Display file information if data is loaded
        if st.session_state.get('data_loaded', False) and st.session_state.get('file_name'):
            st.success(f"✅ File loaded: {st.session_state.file_name}")

            # Show data statistics if available
            if st.session_state.get('df') is not None:
                df = st.session_state.df
                st.info(f"📊 {len(df)} rows × {len(df.columns)} columns")

                # Show memory usage if available
                if hasattr(df, 'memory_usage'):
                    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.text(f"Memory usage: {memory_usage_mb:.2f} MB")
        else:
            # Show upload hint when no data is loaded
            st.info("📤 Upload an Excel file to begin analysis")

            # Show supported file formats
            with st.expander("Supported file formats", expanded=False):
                st.markdown("""
                - Excel files (.xlsx, .xls)
                - Files should contain:
                  - At least one numeric column for analysis
                  - Preferably date columns for time series analysis
                """)

        # No stop button needed

        # Add a separator
        st.markdown("---")

        # Display version information at the bottom of the sidebar
        try:
            # Try to import the version from app.py
            from app import APP_VERSION
            st.markdown(f"<div style='text-align: center; color: gray; font-size: 0.8em;'>Version: {APP_VERSION}</div>", unsafe_allow_html=True)
        except ImportError:
            # Fallback if version is not defined
            st.markdown("<div style='text-align: center; color: gray; font-size: 0.8em;'>Version: Unknown</div>", unsafe_allow_html=True)

        return selected_section
