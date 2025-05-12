# Excel Data Visualizer

A comprehensive Streamlit application for visualizing, analyzing, and forecasting data from Excel files. This tool provides an intuitive interface for data exploration, statistical analysis, and time series forecasting without requiring programming knowledge.

## Features

- **Data Handling**:
  - Upload and process Excel files (.xlsx, .xls)
  - Automatic data type detection and optimization
  - Intelligent date handling for time series analysis

- **Data Exploration**:
  - Interactive data preview with filtering and sorting
  - Comprehensive statistical summaries
  - Automatic missing value detection and handling

- **Advanced Visualizations**:
  - Histograms and distribution plots
  - Enhanced time series charts with trend analysis
  - Area charts and box plots for data distribution
  - Stacked bar charts for categorical comparisons
  - Correlation heatmaps for relationship analysis
  - Interactive scatter plots with regression lines

- **Smart Features**:
  - AI-powered visualization recommendations
  - Automatic chart configuration based on data types
  - Insights generation for each visualization

- **Business Intelligence**:
  - KPI dashboard with period-based comparisons (Half-Half, Monthly, Quarterly, Yearly)
  - Intelligent date handling for accurate year-over-year comparisons
  - Comparative analysis across different dimensions

- **Advanced Analytics**:
  - Statistical modeling with linear regression
  - Interactive prediction capabilities
  - Performance metrics and model evaluation

- **Time Series Forecasting**:
  - ARIMA models for traditional time series forecasting
  - Prophet models for robust forecasting with seasonality
  - Model performance evaluation and comparison

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Local Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd MyDataVisualizer
   ```

2. Create a virtual environment (recommended):
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Streamlit Cloud Deployment

1. Fork this repository to your GitHub account
2. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your forked repository
4. Set the main file path to `app.py`
5. Deploy the app

## Usage

### Running Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your web browser to the URL shown in the terminal (typically http://localhost:8501).

## How to Use

1. **Upload Data**:
   - Click the file uploader in the main panel
   - Select an Excel file (.xlsx or .xls format)
   - Wait for the data to load and process

2. **Navigate Sections**:
   - Use the sidebar menu to switch between different analysis modules:
     - **Data Preview**: View and explore your raw data
     - **Statistical Details**: See summary statistics and distributions
     - **Data Visualization**: Create custom charts and visualizations
     - **Smart Visuals**: Get AI-recommended visualizations
     - **Advanced Analytics**: Perform statistical analysis and modeling
     - **Forecasting**: Generate time series forecasts

3. **Work with Visualizations**:
   - Select columns and chart types using the dropdown menus
   - Adjust parameters to customize your visualizations
   - Hover over charts for interactive tooltips and details
   - Download charts and data for external use

4. **Analyze Data**:
   - Use the KPI dashboard to track key metrics
   - Compare data across different time periods
   - Build statistical models to identify relationships
   - Generate forecasts for future trends

## Special Features

### Date Handling Capabilities

The application includes intelligent date handling capabilities:

- **Hidden Year Detection**: Automatically detects date columns with hidden year components (e.g., dates that visually show only month and day but internally contain the full date)
- **Standardized Date Columns**: Creates standardized date columns with proper year components for accurate year-over-year comparisons
- **Smart KPI Calculations**: Ensures accurate percentage growth calculations even when using date columns with limited visual information

### Performance Optimizations

- **Data Caching**: Frequently used data and calculations are cached for faster performance
- **Lazy Loading**: Components are loaded only when needed to improve initial load time
- **Automatic Downsampling**: Large datasets are intelligently downsampled for visualization while preserving trends

## Deployment Options

### Streamlit Cloud

For the easiest deployment experience, use Streamlit Cloud:

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Configure the app settings (main file: `app.py`)
4. Deploy with a single click

### Docker Deployment

For containerized deployment:

1. Create a Dockerfile in the project root:
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["streamlit", "run", "app.py"]
   ```

2. Build and run the Docker container:
   ```bash
   docker build -t data-visualizer .
   docker run -p 8501:8501 data-visualizer
   ```

## System Requirements

- **Python**: 3.8 or higher
- **Browser**: Chrome, Firefox, Edge, or Safari (latest versions)
- **RAM**: 4GB minimum, 8GB recommended for larger datasets
- **Storage**: 500MB for application, additional space for data files
- **Operating System**: Any OS that supports Python (Windows, macOS, Linux)

## Dependencies

The application relies on the following key libraries:

- **Core Framework**:
  - Streamlit: Web application framework
  - Streamlit-Extras & Streamlit-Option-Menu: Enhanced UI components

- **Data Processing**:
  - Pandas & NumPy: Data manipulation and analysis
  - OpenPyXL: Excel file handling

- **Visualization**:
  - Plotly: Interactive visualizations
  - Matplotlib & Seaborn: Statistical visualizations

- **Analytics & Forecasting**:
  - Scikit-learn: Machine learning and statistical modeling
  - Statsmodels: Statistical modeling and time series analysis
  - Prophet: Time series forecasting with seasonality

## Troubleshooting

If you encounter issues:

1. **Installation Problems**:
   - Ensure you have the correct Python version (3.8+)
   - Try installing dependencies individually if the requirements.txt installation fails
   - For Prophet installation issues, see [Prophet Installation Guide](https://facebook.github.io/prophet/docs/installation.html)

2. **Runtime Errors**:
   - Check that your Excel file is properly formatted
   - Ensure your data has appropriate columns for the analysis you're attempting
   - For large files, try reducing the file size or using data aggregation options

3. **Performance Issues**:
   - Use data aggregation options for large time series
   - Disable rolling averages if not needed
   - Consider filtering your data to focus on specific time periods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
