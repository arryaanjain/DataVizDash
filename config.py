"""
Configuration settings for the application.
"""

# Define color schemes for light and dark mode
LIGHT_THEME = {
    "navy_blue": "#0A2342",
    "gold": "#F5A623",
    "slate_gray": "#708090",
    "light_gray": "#F5F5F5",
    "white": "#FFFFFF",
    "dark_blue": "#001233",
    "background": "#F5F5F5",
    "text": "#0A2342",
    "card_bg": "#FFFFFF",
    "sidebar_bg": "#0A2342",
    "sidebar_text": "#FFFFFF"
}

DARK_THEME = {
    "navy_blue": "#0A2342",
    "gold": "#F5A623",
    "slate_gray": "#708090",
    "light_gray": "#1E1E1E",
    "white": "#2D2D2D",
    "dark_blue": "#001233",
    "background": "#121212",
    "text": "#F5F5F5",
    "card_bg": "#2D2D2D",
    "sidebar_bg": "#0A2342",
    "sidebar_text": "#F5F5F5"
}

# App settings
APP_TITLE = "Data Visualizer"
APP_ICON = "📊"
APP_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Navigation options
NAV_OPTIONS = ["Data Preview", "Statistical Details", "Data Visualization", "Smart Visuals", "Advanced Analytics", "Forecasting"]
NAV_ICONS = ["table", "calculator", "bar-chart-fill", "magic", "graph-up", "calendar-check"]

# Visualization settings
VIZ_TABS = ["Numeric", "Categorical", "Relationships", "Rich Visuals"]
RICH_VIZ_TYPES = ["Pie Chart", "Enhanced Time Series", "Area Chart", "Box Plot", "Stacked Bar Chart", "Side-by-Side Chart"]

# Smart visualization settings
SMART_CHART_TYPES = [
    "Pie Chart",
    "Bar Chart",
    "Time Series",
    "Heatmap",
    "Box Plot",
    "Stacked Bar Chart"
]

# Analytics settings
ANALYTICS_TABS = ["KPI Dashboard", "Comparative Analysis", "Statistical Models"]

# Forecasting settings
FORECAST_TABS = ["ARIMA Forecast", "Prophet Forecast"]
MIN_FORECAST_DATA_POINTS = 24  # Minimum number of data points required for forecasting
