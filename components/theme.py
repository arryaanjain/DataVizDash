"""
Theme and styling components for the application.
"""
import streamlit as st
from config import LIGHT_THEME, DARK_THEME

def get_current_theme():
    """Get the current theme based on dark mode setting."""
    return DARK_THEME if st.session_state.dark_mode else LIGHT_THEME

def apply_custom_css(theme):
    """Apply custom CSS styling based on the current theme."""
    st.markdown(f"""
    <style>
        /* Color palette */
        :root {{
            --navy-blue: {theme["navy_blue"]};
            --gold: {theme["gold"]};
            --slate-gray: {theme["slate_gray"]};
            --light-gray: {theme["light_gray"]};
            --white: {theme["white"]};
            --dark-blue: {theme["dark_blue"]};
            --background: {theme["background"]};
            --text: {theme["text"]};
            --card-bg: {theme["card_bg"]};
        }}

        /* Main background and text */
        .main {{
            background-color: var(--background);
            color: var(--text);
        }}

        /* Sidebar styling */
        .css-1d391kg, .css-1cypcdb {{
            background-color: {theme["sidebar_bg"]};
            color: {theme["sidebar_text"]};
        }}

        /* Headers */
        h1, h2, h3 {{
            color: var(--text);
            font-weight: bold;
        }}

        /* Cards */
        .card {{
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            background-color: var(--card-bg);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}

        /* KPI cards */
        .kpi-card {{
            background-color: var(--navy-blue);
            color: var(--white);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }}

        .kpi-value {{
            font-size: 24px;
            font-weight: bold;
            color: var(--gold);
        }}

        .kpi-label {{
            font-size: 14px;
            color: var(--light-gray);
        }}

        /* Upload area */
        .upload-area {{
            border: 2px dashed var(--slate-gray);
            border-radius: 10px;
            padding: 30px;
            text-align: center;
            background-color: var(--card-bg);
            transition: all 0.3s ease;
        }}

        .upload-area:hover {{
            border-color: var(--gold);
            background-color: rgba(245, 166, 35, 0.05);
        }}

        /* Buttons */
        .stButton>button {{
            background-color: var(--navy-blue);
            color: var(--white);
            border-radius: 5px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            background-color: var(--gold);
            color: var(--navy-blue);
        }}

        /* Tooltip */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}

        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 120px;
            background-color: var(--navy-blue);
            color: var(--white);
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        /* Data table */
        .dataframe {{
            border-collapse: collapse;
            width: 100%;
            border-radius: 5px;
            overflow: hidden;
        }}

        .dataframe th {{
            background-color: var(--navy-blue);
            color: var(--white);
            padding: 12px;
            text-align: left;
        }}

        .dataframe td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
            background-color: var(--card-bg);
            color: var(--text);
        }}

        .dataframe tr:nth-child(even) {{
            background-color: rgba(0, 0, 0, 0.05);
        }}

        .dataframe tr:hover {{
            background-color: rgba(245, 166, 35, 0.05);
        }}

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: var(--card-bg);
            color: var(--text);
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            border: none;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: var(--navy-blue) !important;
            color: var(--white) !important;
        }}

        /* Search box */
        .search-box {{
            padding: 10px;
            border-radius: 5px;
            border: 1px solid var(--slate-gray);
            width: 100%;
            background-color: var(--card-bg);
            color: var(--text);
        }}

        /* Animation for charts */
        .chart-container {{
            transition: all 0.5s ease;
        }}

        .chart-container:hover {{
            transform: scale(1.01);
        }}
    </style>
    """, unsafe_allow_html=True)

def initialize_theme():
    """Initialize the theme settings."""
    # Initialize session state for dark mode if it doesn't exist
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Apply the current theme
    apply_custom_css(get_current_theme())
