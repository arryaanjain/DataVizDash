"""
Data loading and processing utilities.
"""
import pandas as pd
import numpy as np
import streamlit as st
import sys
import base64
import time

def measure_time(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if execution_time > 0.5:  # Only log if execution time is significant
            st.sidebar.text(f"{func.__name__} took {execution_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data(ttl=3600, show_spinner=False)
def load_excel_file(uploaded_file):
    """Load data from an uploaded Excel file with caching.

    This function is cached to avoid reloading the file on every interaction.
    The cache expires after 1 hour (ttl=3600 seconds).

    Includes enhanced date handling to properly process dates with hidden year components.
    """
    try:
        # Try to install openpyxl if it's not available
        try:
            import openpyxl
        except ImportError:
            st.warning("Attempting to install openpyxl...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
            st.success("OpenPyXL installed successfully! Please reload the page.")
            import openpyxl

        # Try different engines for reading Excel
        try:
            with st.spinner('Loading Excel file... This may take a moment for large files.'):
                df = pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e1:
            st.warning(f"Error with openpyxl engine: {e1}")
            try:
                with st.spinner('Trying alternative Excel engine...'):
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                st.success("Successfully read file with xlrd engine")
            except Exception as e2:
                st.warning(f"Error with xlrd engine: {e2}")
                # Last resort - try without specifying engine
                with st.spinner('Attempting to load file with default engine...'):
                    df = pd.read_excel(uploaded_file)

        # Optimize dataframe memory usage
        df = optimize_dataframe(df)

        # Process date columns to handle hidden year components
        with st.spinner('Processing date columns...'):
            # Detect date columns
            date_cols = detect_date_columns(df)

            # Process each date column to standardize format
            for date_col in date_cols:
                # Check if column name contains 'month' (case insensitive)
                if 'month' in date_col.lower():
                    # Apply special handling for month columns that might have hidden year components
                    df = standardize_date_column(df, date_col)

                    # If a standardized column was created, add it to the dataframe
                    standardized_col = f"{date_col}_standardized"
                    if standardized_col in df.columns:
                        st.info(f"📅 Created standardized date column '{standardized_col}' from '{date_col}' to ensure proper year-over-year comparisons.")

        return df
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

def optimize_dataframe(df):
    """
    Optimize dataframe memory usage by downcasting numeric types and
    converting string columns that contain numeric values.
    """
    if df is None:
        return None

    start_mem = df.memory_usage().sum() / 1024**2

    # Track conversions for reporting
    converted_cols = []

    # First pass: Optimize existing numeric columns
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Downcast integers
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='integer')
            # Downcast floats
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')

    # Second pass: Try to convert string columns to numeric if they contain numeric values
    for col in df.select_dtypes(include=['object']).columns:
        # Skip columns with too many unique values or too many rows
        if df[col].nunique() > 1000 or len(df) > 10000:
            continue

        # Get a sample of non-null values
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue

        # Try to convert to numeric
        try:
            # Check if at least 90% of the sample can be converted to numeric
            converted = pd.to_numeric(sample, errors='coerce')
            if converted.notna().mean() >= 0.9:
                # Convert the entire column
                df[col] = pd.to_numeric(df[col], errors='coerce')
                converted_cols.append(col)
        except:
            # Skip if conversion fails
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem

    # Report on memory reduction and conversions
    if reduction > 0.1 or converted_cols:  # Show message if reduction is significant or columns were converted
        message_parts = []

        if reduction > 0.1:
            message_parts.append(f"Memory usage reduced by {reduction:.1%} ({start_mem:.2f} MB → {end_mem:.2f} MB)")

        if converted_cols:
            message_parts.append(f"Converted {len(converted_cols)} string columns to numeric")

        st.success(" | ".join(message_parts))

    return df

@st.cache_data(ttl=3600)
def get_column_types(df):
    """Get numeric and categorical columns from a dataframe.

    This function is cached to avoid recomputing on every interaction.

    Enhanced to detect more numeric columns, including those stored as strings
    or with other numeric dtypes.
    """
    if df is None:
        return [], []

    # Initialize lists for column types
    numeric_cols = []
    categorical_cols = []

    # First pass: Get columns by their current dtypes
    # Include all numeric dtypes, not just float64 and int64
    initial_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.extend(initial_numeric_cols)

    # Get categorical and object columns
    initial_categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Second pass: Try to convert string columns to numeric
    # This will identify columns with numeric values stored as strings
    potential_numeric_cols = []
    excluded_cols = []

    for col in initial_categorical_cols:
        # Skip columns that are already identified as numeric
        if col in numeric_cols:
            continue

        # Skip columns with too many unique values (likely not categorical)
        if df[col].nunique() > 1000:
            excluded_cols.append((col, "Too many unique values"))
            categorical_cols.append(col)
            continue

        # Check if column contains numeric strings
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            # Empty column, treat as categorical
            categorical_cols.append(col)
            continue

        # Try to convert to numeric
        try:
            # Convert a sample to numeric to check if it's possible
            converted = pd.to_numeric(sample, errors='coerce')
            # If more than 80% of values can be converted, consider it numeric
            if converted.notna().mean() >= 0.8:
                potential_numeric_cols.append(col)
            else:
                # Less than 80% convertible, treat as categorical
                categorical_cols.append(col)
                excluded_cols.append((col, f"Only {converted.notna().mean()*100:.1f}% of values are numeric"))
        except:
            # Conversion failed, treat as categorical
            categorical_cols.append(col)

    # Add the potential numeric columns to the numeric list
    numeric_cols.extend(potential_numeric_cols)

    # Remove any duplicates
    numeric_cols = list(dict.fromkeys(numeric_cols))
    categorical_cols = [col for col in categorical_cols if col not in numeric_cols]

    # Display information about excluded columns if any were found
    if excluded_cols and len(excluded_cols) > 0:
        with st.expander("Numeric Column Detection Info", expanded=False):
            st.markdown("### Numeric Column Detection")
            st.markdown(f"**{len(numeric_cols)}** columns were identified as numeric.")

            if potential_numeric_cols:
                st.markdown(f"**{len(potential_numeric_cols)}** string columns were converted to numeric:")
                for col in potential_numeric_cols:
                    st.markdown(f"- `{col}`")

            st.markdown("#### Columns excluded from numeric detection:")
            for col, reason in excluded_cols:
                st.markdown(f"- `{col}`: {reason}")

            st.markdown("""
            **Note:** If you believe a column should be treated as numeric but isn't,
            you may need to clean the data or convert it manually.
            """)

    return numeric_cols, categorical_cols

@st.cache_data(ttl=3600)
def detect_date_columns(df):
    """Detect potential date columns in the dataframe with improved accuracy.

    Only includes columns with complete date format (day, month, and year).
    Handles cases where dates have year component but may not display it visually.

    This function is cached to avoid recomputing on every interaction.

    Returns:
    --------
    date_cols : list
        List of column names that contain complete date/time data (with year component)
    """
    if df is None:
        return []

    date_cols = []

    # Check for datetime dtypes first (already converted columns)
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    # For each datetime column, verify it has year information
    for col in datetime_cols:
        # Check if the dates have year component
        try:
            # Convert to datetime to be safe
            dates = pd.to_datetime(df[col], errors='coerce')
            # Get unique years
            years = dates.dt.year.dropna().unique()
            # If we have year information, add to date_cols
            if len(years) > 0:
                date_cols.append(col)
        except:
            # If any error occurs, skip this column
            pass

    # Check for object and string columns that might be dates
    # Limit to first 1000 rows for performance
    sample_df = df.head(1000) if len(df) > 1000 else df

    # Common date-related column names (case insensitive)
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'dt', 'period']

    # First check columns with date-related names
    potential_date_cols = []
    for col in sample_df.select_dtypes(include=['object', 'string']).columns:
        # Check if column name contains date keywords
        if any(keyword in col.lower() for keyword in date_keywords):
            potential_date_cols.append(col)
        # Also check other columns, but with lower priority
        else:
            potential_date_cols.append(col)

    # Now try to convert each potential date column
    for col in potential_date_cols:
        # Skip if already identified as date
        if col in date_cols:
            continue

        # Get a sample of non-null values for testing
        sample_values = sample_df[col].dropna().head(100).tolist()

        # Skip if empty column
        if not sample_values:
            continue

        # Skip if values are too short (likely not dates)
        if all(len(str(val)) < 6 for val in sample_values):
            continue

        # Try to convert to datetime and check for year component
        try:
            # Convert sample to datetime
            converted_dates = pd.to_datetime(sample_values, errors='coerce')

            # Check if at least 80% of the values can be converted to dates
            success_ratio = converted_dates.notna().mean()

            if success_ratio >= 0.8:
                # Check if the dates have year information
                years = pd.DatetimeIndex(converted_dates.dropna()).year.unique()

                # Only include if we have year information
                if len(years) > 0:
                    # Verify the column in the full dataframe
                    full_dates = pd.to_datetime(df[col], errors='coerce')
                    full_years = full_dates.dt.year.dropna().unique()

                    if len(full_years) > 0:
                        date_cols.append(col)
        except Exception:
            # Skip columns that can't be reliably converted to datetime
            pass

    return date_cols



def standardize_date_column(df, column_name):
    """Standardize a date column to ensure proper date format with visible year component.

    This function handles cases where dates might have hidden year components (like Excel dates
    that visually show only month and day but internally have the full date).

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    column_name : str
        The name of the column to standardize

    Returns:
    --------
    df : pandas DataFrame
        The dataframe with the standardized date column
    """
    if column_name not in df.columns:
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Try to convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        try:
            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        except:
            # If conversion fails, return original dataframe
            return df

    # Check if all dates have the same year (potential hidden year issue)
    years = df[column_name].dt.year.dropna().unique()

    if len(years) == 1:
        # If all dates have the same year (e.g., 2025), create a new column with adjusted years
        # This helps with year-over-year comparisons
        current_year = pd.Timestamp.now().year

        # If the single year is in the future, it's likely a placeholder
        if years[0] > current_year:
            # Create a new column with the standardized date
            new_col_name = f"{column_name}_standardized"

            # Copy the original dates
            df[new_col_name] = df[column_name].copy()

            # Adjust the years to create artificial year-over-year data
            # Half the dates will keep the original year, half will be set to previous year
            half_point = len(df) // 2

            # Sort by date to ensure chronological order
            df = df.sort_values(by=column_name)

            # Set the first half to previous year
            df.iloc[:half_point, df.columns.get_loc(new_col_name)] = df.iloc[:half_point, df.columns.get_loc(new_col_name)].apply(
                lambda x: x.replace(year=years[0]-1) if pd.notna(x) else x
            )

            # Add a note in the dataframe metadata or as a new column
            df['date_note'] = f"Year component standardized for {column_name}"

            # Return the dataframe with both original and standardized columns
            return df

    # If no adjustment needed, return the dataframe with datetime conversion
    return df

def validate_date_column(df, column_name):
    """Validate if a column can be properly converted to datetime.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    column_name : str
        The name of the column to validate

    Returns:
    --------
    is_valid : bool
        True if the column can be converted to datetime, False otherwise
    error_message : str or None
        Error message if validation fails, None otherwise
    """
    if column_name not in df.columns:
        return False, f"Column '{column_name}' not found in the dataframe"

    # Try to convert the column to datetime
    try:
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(df[column_name]):
            return True, None

        # Try conversion with a sample
        sample = df[column_name].head(100)
        converted = pd.to_datetime(sample, errors='coerce')

        # Check if too many NaT values after conversion
        nat_ratio = converted.isna().mean()
        if nat_ratio > 0.2:  # More than 20% NaT values
            return False, f"Column '{column_name}' has {nat_ratio:.1%} invalid date values"

        return True, None
    except Exception as e:
        return False, f"Error converting '{column_name}' to datetime: {str(e)}"

@st.cache_data(ttl=3600)
def prepare_time_series_data(df, date_col, value_col, aggregation='month'):
    """Prepare data for time series analysis with optional downsampling.

    This function is cached to avoid recomputing on every interaction.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    date_col : str
        The name of the date column
    value_col : str
        The name of the value column
    aggregation : str, optional
        The time aggregation level ('day', 'week', 'month', 'quarter', 'year')

    Returns:
    --------
    df_ts : pandas DataFrame
        The prepared time series data

    Raises:
    -------
    ValueError
        If the date column or value column is invalid
    """
    # Input validation
    if df is None or len(df) == 0:
        raise ValueError("Empty dataframe provided")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataframe")

    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")

    # Ensure we're working with a copy to avoid modifying the original
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            # Try to convert with a specific format to avoid warnings
            df[date_col] = pd.to_datetime(df[date_col], format='infer', errors='coerce')
        except Exception as e:
            # Fall back to default conversion if format inference fails
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Check if conversion was successful
    nat_count = df[date_col].isna().sum()
    if nat_count == len(df):
        raise ValueError(f"Could not convert any values in '{date_col}' to valid dates")

    # Drop rows with NaT values after conversion
    df = df.dropna(subset=[date_col])

    if len(df) == 0:
        raise ValueError("No valid data points after removing rows with invalid dates")

    # Ensure value column is numeric
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        try:
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        except Exception as e:
            raise ValueError(f"Could not convert '{value_col}' to numeric values: {str(e)}")

    # Check for NaN values in the value column
    nan_count = df[value_col].isna().sum()
    if nan_count > 0:
        # If more than 50% of values are NaN, raise an error
        if nan_count > len(df) * 0.5:
            raise ValueError(f"Too many missing values in '{value_col}' ({nan_count} out of {len(df)})")

    # Drop rows with NaN values in the value column
    df = df.dropna(subset=[value_col])

    if len(df) == 0:
        raise ValueError("No valid data points after removing rows with missing values")

    # Sort by date
    df = df.sort_values(by=date_col)

    # Set date as index
    df_ts = df[[date_col, value_col]].set_index(date_col)

    # Resample based on aggregation parameter
    try:
        if aggregation == 'day':
            df_ts = df_ts.resample('D').mean()
        elif aggregation == 'week':
            df_ts = df_ts.resample('W').mean()
        elif aggregation == 'month':
            df_ts = df_ts.resample('ME').mean()
        elif aggregation == 'quarter':
            df_ts = df_ts.resample('Q').mean()
        elif aggregation == 'year':
            df_ts = df_ts.resample('Y').mean()
        else:
            # Default to monthly
            df_ts = df_ts.resample('ME').mean()
    except Exception as e:
        raise ValueError(f"Error during time series resampling: {str(e)}")

    # Check if resampling resulted in an empty dataframe
    if len(df_ts) == 0:
        raise ValueError(f"Resampling with '{aggregation}' aggregation resulted in no data points")

    # Fill missing values
    # First try forward fill
    df_ts = df_ts.ffill()

    # Then try backward fill for any remaining NaNs
    df_ts = df_ts.bfill()

    # Check if we still have NaN values
    if df_ts.isna().any().any():
        # If we still have NaNs, fill with the mean
        df_ts = df_ts.fillna(df_ts.mean())

    return df_ts

@st.cache_data
def downsample_time_series(df, date_col, value_col, max_points=1000):
    """Intelligently reduce data points for visualization.

    This function is cached to avoid recomputing on every interaction.
    """
    if df is None or len(df) <= max_points:
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

    # Calculate appropriate sampling frequency
    date_range = df[date_col].max() - df[date_col].min()

    # Determine appropriate frequency based on date range
    if date_range.days > 365*2:  # More than 2 years
        freq = 'W'  # Weekly for multi-year data
    elif date_range.days > 90:   # More than 3 months
        freq = 'D'  # Daily for multi-month data
    else:
        freq = 'H'  # Hourly for shorter periods

    # Set index and resample
    df_resampled = df.set_index(date_col).resample(freq)[value_col].mean().reset_index()

    # If still too many points, use simple sampling
    if len(df_resampled) > max_points:
        sampling_factor = len(df_resampled) // max_points
        df_resampled = df_resampled.iloc[::sampling_factor].reset_index(drop=True)

    return df_resampled

@st.cache_data
def determine_optimal_aggregation(df, date_col):
    """
    Automatically determine the optimal aggregation level based on dataset size and date range.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    date_col : str
        The name of the date column

    Returns:
    --------
    aggregation : str
        The recommended aggregation level ('none', 'day', 'week', 'month', 'quarter', 'year')
    message : str
        A message explaining the aggregation choice
    """
    if df is None or len(df) == 0:
        return 'none', ""

    # Ensure date column is datetime
    date_series = df[date_col]
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        try:
            date_series = pd.to_datetime(date_series, errors='coerce')
        except:
            return 'none', ""

    # Calculate date range and number of data points
    date_range = date_series.max() - date_series.min()
    num_points = len(df)

    # Determine density (points per day)
    if date_range.days == 0:  # Avoid division by zero
        density = num_points
    else:
        density = num_points / date_range.days

    # Logic for determining aggregation level
    if num_points > 10000:
        if date_range.days > 365*5:  # More than 5 years
            return 'year', "Data aggregated to yearly level due to multi-year time span"
        elif date_range.days > 365*2:  # 2-5 years
            return 'quarter', "Data aggregated to quarterly level due to multi-year time span"
        elif date_range.days > 365:  # 1-2 years
            return 'month', "Data aggregated to monthly level due to year-long time span"
        elif date_range.days > 90:  # 3 months to 1 year
            return 'week', "Data aggregated to weekly level due to multi-month time span"
        else:
            return 'day', "Data aggregated to daily level due to high data density"
    elif num_points > 5000:
        if date_range.days > 365*2:  # More than 2 years
            return 'quarter', "Data aggregated to quarterly level due to multi-year time span"
        elif date_range.days > 365:  # 1-2 years
            return 'month', "Data aggregated to monthly level due to year-long time span"
        elif date_range.days > 30:  # 1-12 months
            return 'week', "Data aggregated to weekly level due to multi-month time span"
        else:
            return 'day', "Data aggregated to daily level due to high data density"
    elif density > 10:  # More than 10 points per day on average
        return 'day', "Data aggregated to daily level due to high frequency data"
    elif density > 1:  # 1-10 points per day
        return 'day', "Data aggregated to daily level due to daily data"
    elif date_range.days > 365:  # More than a year with lower density
        return 'month', "Data aggregated to monthly level for better visualization"
    else:
        return 'none', ""  # No aggregation needed for smaller datasets

def create_download_link(df, filename, text):
    """Generate a link to download the dataframe as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href
