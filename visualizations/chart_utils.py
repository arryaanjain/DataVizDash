"""
Chart utility functions for creating various visualizations.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from utils.data_utils import downsample_time_series, measure_time

@st.cache_data(ttl=3600)
def create_pie_chart(df, column):
    """Create a pie chart for categorical data with caching."""
    # Handle missing values and ensure we're working with a copy
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[column])

    # Get value counts
    value_counts = df_clean[column].value_counts()

    # Validate data - ensure we have values to plot
    if len(value_counts) == 0:
        # Create an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for pie chart",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig

    # If there are too many categories, keep only top 5 and group others
    if len(value_counts) > 5:
        top_5 = value_counts.head(5)
        others = pd.Series({'Others': value_counts[5:].sum()})
        value_counts = pd.concat([top_5, others])

    # Create a DataFrame for plotly express (more reliable than passing Series)
    pie_df = pd.DataFrame({
        'Category': value_counts.index,
        'Value': value_counts.values
    })

    # Calculate total for percentage validation
    total = pie_df['Value'].sum()

    # Create the pie chart using the DataFrame
    fig = px.pie(
        pie_df,
        names='Category',
        values='Value',
        title=f'Distribution of {column}',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3,  # Make it a donut chart
    )

    # Add value labels with percentages and absolute values
    # Use custom hovertemplate to ensure correct percentage calculation
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',  # Simplified text info to avoid overlap
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}',
        insidetextfont=dict(size=10)
    )

    # Add a separate trace for value labels if there aren't too many segments
    if len(value_counts) <= 8:
        # Add value annotations to each segment
        for i, row in enumerate(pie_df.itertuples()):
            value = row.Value

            # Add value annotation
            fig.add_annotation(
                text=f"Value: {value}",
                x=0.5 + 0.4 * np.cos(2 * np.pi * (i / len(pie_df) + 0.25)),
                y=0.5 + 0.4 * np.sin(2 * np.pi * (i / len(pie_df) + 0.25)),
                showarrow=False,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255, 255, 255, 0.7)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                borderpad=2
            )

    # Improve layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=50, b=100, l=20, r=20),
        annotations=[
            dict(
                text=f"Total: {total:.1f}",
                x=0.5, y=-0.15,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )

    return fig

def create_stacked_bar(df, x_col, y_col, color_col):
    """Create a stacked bar chart."""
    # Sort data for consistent presentation
    df_sorted = df.sort_values(by=[x_col, color_col])

    fig = px.bar(
        df_sorted,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f'{y_col} by {x_col} and {color_col}',
        barmode='stack',
        text=y_col  # Add value labels
    )

    # Format the text labels
    fig.update_traces(
        texttemplate='%{y:.2f}',
        textposition='inside',
        insidetextfont=dict(color='white', size=10)
    )

    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=color_col,
        margin=dict(t=50, b=100, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    # Add summary annotation
    total_value = df[y_col].sum()
    fig.add_annotation(
        text=f"Total {y_col}: {total_value:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    # Ensure x-axis labels are properly aligned
    if len(df[x_col].unique()) > 5:
        fig.update_xaxes(tickangle=-45)

    return fig

@st.cache_data(ttl=3600)
@measure_time
def create_line_chart(df, x_col, y_col, color_col=None):
    """Create a line chart for trend analysis with caching and performance optimizations."""
    # Check if we need to downsample for performance
    MAX_POINTS = 5000

    # For datetime x-axis, use specialized downsampling
    is_datetime = False
    try:
        if pd.api.types.is_datetime64_any_dtype(df[x_col]):
            is_datetime = True
        else:
            # Try to convert to datetime
            pd.to_datetime(df[x_col])
            is_datetime = True
    except:
        pass

    # Downsample if needed
    if len(df) > MAX_POINTS:
        if is_datetime:
            # Use time series downsampling for datetime x-axis
            if color_col:
                # Downsample each category separately
                downsampled_dfs = []
                for category in df[color_col].unique():
                    category_df = df[df[color_col] == category]
                    # Calculate points per category
                    category_max_points = max(100, int(MAX_POINTS * len(category_df) / len(df)))
                    downsampled_category = downsample_time_series(
                        category_df,
                        x_col,
                        y_col,
                        max_points=category_max_points
                    )
                    downsampled_dfs.append(downsampled_category)
                plot_df = pd.concat(downsampled_dfs)
            else:
                plot_df = downsample_time_series(df, x_col, y_col, max_points=MAX_POINTS)
        else:
            # For non-datetime, use simple sampling
            sampling_factor = len(df) // MAX_POINTS
            plot_df = df.iloc[::sampling_factor].copy()
    else:
        plot_df = df

    # Determine if we should use markers based on data size
    use_markers = len(plot_df) < 100

    # For large datasets, use more efficient Graph Objects
    if len(plot_df) > 1000:
        fig = go.Figure()

        if color_col:
            # Create a trace for each category
            for category in plot_df[color_col].unique():
                category_df = plot_df[plot_df[color_col] == category]
                fig.add_trace(go.Scatter(
                    x=category_df[x_col],
                    y=category_df[y_col],
                    mode='lines' if not use_markers else 'lines+markers',
                    name=str(category),
                    line=dict(width=2),
                    marker=dict(size=4) if use_markers else None
                ))
        else:
            # Create a single trace
            fig.add_trace(go.Scatter(
                x=plot_df[x_col],
                y=plot_df[y_col],
                mode='lines' if not use_markers else 'lines+markers',
                name=y_col,
                line=dict(width=2),
                marker=dict(size=4) if use_markers else None
            ))

        # Add title
        fig.update_layout(title=f'Trend of {y_col} over {x_col}')
    else:
        # For smaller datasets, use Plotly Express
        if color_col:
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f'Trend of {y_col} over {x_col}',
                markers=use_markers
            )
        else:
            fig = px.line(
                plot_df,
                x=x_col,
                y=y_col,
                title=f'Trend of {y_col} over {x_col}',
                markers=use_markers
            )

    # Add value labels at key points (min, max, last point)
    # Only add for datasets that aren't too large to avoid clutter
    if len(plot_df) < 300:
        # Find key points
        if color_col:
            # Handle each category separately
            for category in plot_df[color_col].unique():
                category_df = plot_df[plot_df[color_col] == category]

                # Get key points
                max_idx = category_df[y_col].idxmax()
                min_idx = category_df[y_col].idxmin()
                last_idx = category_df.index[-1]

                # Create unique indices list (in case min/max/last are the same point)
                key_indices = list(set([max_idx, min_idx, last_idx]))
                key_points = category_df.loc[key_indices]

                # Add labels
                fig.add_trace(go.Scatter(
                    x=key_points[x_col],
                    y=key_points[y_col],
                    mode='text',
                    text=[f"{y:.2f}" for y in key_points[y_col]],
                    textposition='top center',
                    textfont=dict(size=10),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        else:
            # Get key points for the entire dataset
            try:
                max_idx = plot_df[y_col].idxmax()
                min_idx = plot_df[y_col].idxmin()
                last_idx = plot_df.index[-1]

                # Create unique indices list (in case min/max/last are the same point)
                key_indices = list(set([max_idx, min_idx, last_idx]))
                key_points = plot_df.loc[key_indices]

                # Add labels
                fig.add_trace(go.Scatter(
                    x=key_points[x_col],
                    y=key_points[y_col],
                    mode='text',
                    text=[f"{y:.2f}" for y in key_points[y_col]],
                    textposition='top center',
                    textfont=dict(size=10),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            except:
                # Skip if there's an error (e.g., with index access)
                pass

    # Add summary statistics annotation
    avg_value = plot_df[y_col].mean()
    max_value = plot_df[y_col].max()
    min_value = plot_df[y_col].min()

    # Optimize layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(l=0, r=0, t=30, b=50),  # Adjust bottom margin for annotation
        hovermode="closest",
        uirevision='constant',  # Preserve UI state on updates
        template="plotly_white",  # Use a simpler template
        annotations=[
            dict(
                text=f"Avg: {avg_value:.2f} | Max: {max_value:.2f} | Min: {min_value:.2f}",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color="gray")
            )
        ]
    )

    # Add range slider for datetime x-axis
    if is_datetime and len(plot_df) < 10000:
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

    return fig

def create_area_chart(df, x_col, y_col, **kwargs):
    """Create an area chart with value labels and summary.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    x_col : str
        The name of the x-axis column
    y_col : str
        The name of the y-axis column
    **kwargs : dict
        Additional parameters (for backward compatibility)
    """

    # Sort data for consistent presentation
    df_sorted = df.sort_values(by=x_col)

    # Determine if we should add markers based on data size
    use_markers = len(df_sorted) < 100

    # No color grouping - removed as requested
    fig = px.area(
        df_sorted,
        x=x_col,
        y=y_col,
        title=f'Area Chart of {y_col} over {x_col}',
        line_shape='spline',  # Smoother lines
        markers=use_markers
    )

    # Add value labels at key points (not all points to avoid clutter)
    if len(df_sorted) < 20:
        # For small datasets, add labels to all points
        fig.add_trace(go.Scatter(
            x=df_sorted[x_col],
            y=df_sorted[y_col],
            mode='text',
            text=[f"{y:.2f}" for y in df_sorted[y_col]],
            textposition='top center',
            textfont=dict(size=10),
            showlegend=False
        ))
    else:
        # For larger datasets, add labels only to min, max, and last points
        # Get key points
        max_idx = df_sorted[y_col].idxmax()
        min_idx = df_sorted[y_col].idxmin()
        last_idx = df_sorted.index[-1]

        key_points = df_sorted.loc[[min_idx, max_idx, last_idx]].drop_duplicates()

        fig.add_trace(go.Scatter(
            x=key_points[x_col],
            y=key_points[y_col],
            mode='text',
            text=[f"{y:.2f}" for y in key_points[y_col]],
            textposition='top center',
            textfont=dict(size=10),
            showlegend=False
        ))

    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(t=50, b=100, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        hovermode="x unified"
    )

    # Add summary statistics
    avg_value = df[y_col].mean()
    max_value = df[y_col].max()
    min_value = df[y_col].min()

    fig.add_annotation(
        text=f"Avg: {avg_value:.2f} | Max: {max_value:.2f} | Min: {min_value:.2f}",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=12, color="gray")
    )

    # Ensure x-axis labels are properly aligned
    if len(df[x_col].unique()) > 5:
        fig.update_xaxes(tickangle=-45)

    return fig

def create_box_plot(df, x_col, y_col):
    """Create a box plot with enhanced value labels and comprehensive summary statistics."""
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=f'Box Plot of {y_col} by {x_col}',
        points='all',  # Show all points
        notched=True   # Show confidence interval around median
    )

    # Add median labels to each box
    categories = df[x_col].unique()
    annotations = []

    for i, category in enumerate(categories):
        category_data = df[df[x_col] == category][y_col]
        median = category_data.median()
        q1 = category_data.quantile(0.25)
        q3 = category_data.quantile(0.75)
        mean = category_data.mean()

        # Calculate IQR for annotations
        iqr = q3 - q1

        # Add median value label with improved visibility
        annotations.append(dict(
            x=i,
            y=median,
            text=f"Median: {median:.2f}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
            font=dict(size=11, color="black", family="Arial Black"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            borderpad=4
        ))

        # Add mean value label
        annotations.append(dict(
            x=i,
            y=mean,
            text=f"Mean: {mean:.2f}",
            showarrow=True,
            arrowhead=7,
            ax=30,
            ay=-20,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)"
        ))

        # Add IQR annotation with improved formatting
        annotations.append(dict(
            x=i,
            y=q3 + (q3-q1)*0.5,  # Position above the box
            text=f"Q1: {q1:.2f}<br>Q3: {q3:.2f}<br>IQR: {iqr:.2f}",
            showarrow=False,
            font=dict(size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            borderpad=4
        ))

    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(t=50, b=120, l=20, r=20),  # Increased bottom margin for summary
        annotations=annotations,
        boxmode='group',
        hovermode='closest'
    )

    # Add comprehensive summary section at the bottom
    summary_text = []
    for category in categories:
        category_data = df[df[x_col] == category][y_col]
        # Calculate key statistics
        median = category_data.median()
        mean = category_data.mean()
        std = category_data.std()
        count = len(category_data)

        summary_text.append(
            f"<b>{category}</b>: Count={count}, Med={median:.2f}, "
            f"Mean={mean:.2f}, SD={std:.2f}, "
            f"Min={category_data.min():.2f}, Max={category_data.max():.2f}"
        )

    # Add a title to the summary section
    summary_header = "<b>Summary Statistics:</b><br>"

    # Combine header with summary text, limiting to first 3 categories if there are many
    if len(summary_text) > 3:
        display_text = summary_header + "<br>".join(summary_text[:3]) + "<br>..."
    else:
        display_text = summary_header + "<br>".join(summary_text)

    fig.add_annotation(
        text=display_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=11, color="black"),
        align="center",
        bgcolor="rgba(240, 240, 240, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=5
    )

    # Ensure x-axis labels are properly aligned
    if len(categories) > 5:
        fig.update_xaxes(tickangle=-45)

    return fig

def create_comparative_chart(df, x_col, y_cols, chart_type='bar', aggregation_method='mean'):
    """Create a comparative chart for multiple metrics with enhanced value labels and summary statistics.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    x_col : str
        The name of the x-axis column
    y_cols : list
        List of column names to plot on the y-axis
    chart_type : str, optional
        The type of chart to create ('bar' or 'line')
    aggregation_method : str, optional
        The method to use for aggregating values ('mean', 'sum', 'min', 'max')
    """
    # Validate aggregation method
    valid_agg_methods = ['mean', 'sum', 'min', 'max']
    if aggregation_method not in valid_agg_methods:
        aggregation_method = 'mean'  # Default to mean if invalid

    if chart_type == 'bar':
        fig = go.Figure()

        # Sort data by x_col to ensure consistent ordering
        df_sorted = df.sort_values(by=x_col)

        # Get unique x values to ensure consistent spacing
        x_values = df_sorted[x_col].unique()

        # Create a color scale for better visual distinction
        colors = px.colors.qualitative.G10[:len(y_cols)]

        # Calculate summary statistics for each column
        summary_stats = {}
        for col in y_cols:
            summary_stats[col] = {
                'mean': df_sorted[col].mean(),
                'max': df_sorted[col].max(),
                'min': df_sorted[col].min(),
                'sum': df_sorted[col].sum()
            }

        for i, col in enumerate(y_cols):
            # Use the selected aggregation method
            if aggregation_method == 'mean':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].mean() for x_val in x_values]
            elif aggregation_method == 'sum':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].sum() for x_val in x_values]
            elif aggregation_method == 'min':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].min() for x_val in x_values]
            elif aggregation_method == 'max':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].max() for x_val in x_values]

            fig.add_trace(go.Bar(
                x=x_values,
                y=y_values,
                name=col,
                marker_color=colors[i],
                text=[f"{y:.2f}" for y in y_values],
                textposition='outside',  # Position text outside bars for better visibility
                textfont=dict(
                    size=11,
                    color='black'
                ),
                texttemplate='%{text}',  # Use formatted text
                outsidetextfont=dict(color=colors[i])  # Match text color to bar color
            ))

        # Add a more descriptive title with aggregation method
        agg_method_display = aggregation_method.upper()
        title = f'Comparison of {", ".join(y_cols)} by {x_col} ({agg_method_display})'

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=f'Value ({agg_method_display})',
            barmode='group',
            bargap=0.15,       # Gap between bars of adjacent location coordinates
            bargroupgap=0.1,   # Gap between bars of the same location coordinates
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,  # Moved down to make room for summary
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=120, l=20, r=20),  # Increased bottom margin for summary
        )

        # Ensure x-axis labels are properly aligned and not overlapping
        fig.update_xaxes(
            tickangle=-45 if len(x_values) > 5 else 0,
            tickmode='array',
            tickvals=list(range(len(x_values))),
            ticktext=[str(x) for x in x_values]
        )

        # Add summary statistics annotation with highlighted selected method
        summary_text = "<b>Summary Statistics:</b><br>"
        for col in y_cols:
            stats = summary_stats[col]

            # Format each statistic, highlighting the selected aggregation method
            mean_text = f"<b>Mean={stats['mean']:.2f}</b>" if aggregation_method == 'mean' else f"Mean={stats['mean']:.2f}"
            sum_text = f"<b>Sum={stats['sum']:.2f}</b>" if aggregation_method == 'sum' else f"Sum={stats['sum']:.2f}"
            min_text = f"<b>Min={stats['min']:.2f}</b>" if aggregation_method == 'min' else f"Min={stats['min']:.2f}"
            max_text = f"<b>Max={stats['max']:.2f}</b>" if aggregation_method == 'max' else f"Max={stats['max']:.2f}"

            summary_text += f"<b>{col}</b>: {mean_text}, {max_text}, {min_text}, {sum_text}<br>"

        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=11, color="black"),
            align="center",
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=5
        )

    elif chart_type == 'line':
        fig = go.Figure()

        # Sort data by x_col to ensure consistent ordering
        df_sorted = df.sort_values(by=x_col)

        # Get unique x values to ensure consistent spacing
        x_values = df_sorted[x_col].unique()

        # Create a color scale for better visual distinction
        colors = px.colors.qualitative.G10[:len(y_cols)]

        # Calculate summary statistics for each column
        summary_stats = {}
        for col in y_cols:
            summary_stats[col] = {
                'mean': df_sorted[col].mean(),
                'max': df_sorted[col].max(),
                'min': df_sorted[col].min(),
                'sum': df_sorted[col].sum()
            }

        for i, col in enumerate(y_cols):
            # Use the selected aggregation method
            if aggregation_method == 'mean':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].mean() for x_val in x_values]
            elif aggregation_method == 'sum':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].sum() for x_val in x_values]
            elif aggregation_method == 'min':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].min() for x_val in x_values]
            elif aggregation_method == 'max':
                y_values = [df_sorted[df_sorted[x_col] == x_val][col].max() for x_val in x_values]

            # Add main line trace
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',  # Remove text from main trace to avoid overlap
                name=col,
                line=dict(color=colors[i], width=2),
                marker=dict(size=8, line=dict(width=1, color='white'))
            ))

            # Add separate trace for text labels with improved visibility
            # Only add labels for min, max, and last points to avoid clutter
            if len(y_values) > 0:
                max_idx = y_values.index(max(y_values))
                min_idx = y_values.index(min(y_values))
                last_idx = len(y_values) - 1

                # Create unique indices list
                key_indices = list(set([max_idx, min_idx, last_idx]))

                # Extract key points
                key_x = [x_values[idx] for idx in key_indices]
                key_y = [y_values[idx] for idx in key_indices]

                # Add text labels for key points
                fig.add_trace(go.Scatter(
                    x=key_x,
                    y=key_y,
                    mode='text',
                    text=[f"{y:.2f}" for y in key_y],
                    textposition='top center',
                    textfont=dict(
                        size=11,
                        color=colors[i],
                        family="Arial"
                    ),
                    showlegend=False,
                    hoverinfo='skip',
                    texttemplate='%{text}'
                ))

        # Add a more descriptive title with aggregation method
        agg_method_display = aggregation_method.upper()
        title = f'Trend Comparison of {", ".join(y_cols)} by {x_col} ({agg_method_display})'

        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=f'Value ({agg_method_display})',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,  # Moved down to make room for summary
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=50, b=120, l=20, r=20),  # Increased bottom margin for summary
            hovermode="x unified"
        )

        # Ensure x-axis labels are properly aligned and not overlapping
        fig.update_xaxes(
            tickangle=-45 if len(x_values) > 5 else 0
        )

        # Add summary statistics annotation with highlighted selected method
        summary_text = "<b>Summary Statistics:</b><br>"
        for col in y_cols:
            stats = summary_stats[col]

            # Format each statistic, highlighting the selected aggregation method
            mean_text = f"<b>Mean={stats['mean']:.2f}</b>" if aggregation_method == 'mean' else f"Mean={stats['mean']:.2f}"
            sum_text = f"<b>Sum={stats['sum']:.2f}</b>" if aggregation_method == 'sum' else f"Sum={stats['sum']:.2f}"
            min_text = f"<b>Min={stats['min']:.2f}</b>" if aggregation_method == 'min' else f"Min={stats['min']:.2f}"
            max_text = f"<b>Max={stats['max']:.2f}</b>" if aggregation_method == 'max' else f"Max={stats['max']:.2f}"

            summary_text += f"<b>{col}</b>: {mean_text}, {max_text}, {min_text}, {sum_text}<br>"

        fig.add_annotation(
            text=summary_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.18,
            showarrow=False,
            font=dict(size=11, color="black"),
            align="center",
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=5
        )

    return fig

@st.cache_data(ttl=3600, show_spinner=False)
@measure_time
def create_enhanced_time_series_chart(df, date_col, value_col, color_col=None,
                                     aggregation='none', rolling_window=None,
                                     remove_outliers=False, outlier_threshold=3,
                                     date_range=None, selected_categories=None):
    """
    Create an enhanced time series chart with options for aggregation, smoothing, and outlier removal.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the data
    date_col : str
        The name of the date column
    value_col : str
        The name of the value column to plot
    color_col : str, optional
        [DEPRECATED] This parameter is no longer used as color grouping has been removed
    aggregation : str, optional
        The time aggregation level ('none', 'day', 'week', 'month', 'quarter', 'year')
    rolling_window : int, optional
        The window size for rolling average smoothing
    remove_outliers : bool, optional
        Whether to remove outliers from the data
    outlier_threshold : float, optional
        The z-score threshold for outlier detection
    date_range : tuple, optional
        A tuple of (start_date, end_date) to filter the data by date range
    selected_categories : list, optional
        [DEPRECATED] This parameter is no longer used as color grouping has been removed

    Returns:
    --------
    fig : plotly Figure
        The plotly figure object
    """
    # Color grouping has been removed - override parameters
    color_col = None
    selected_categories = None

    # Check if dataframe is too large and needs downsampling
    MAX_POINTS = 5000  # Maximum number of points for efficient rendering

    # Make a copy of the dataframe to avoid modifying the original
    # For large dataframes, only copy the necessary columns to save memory
    if len(df) > MAX_POINTS:
        needed_cols = [date_col, value_col]
        if color_col is not None:
            needed_cols.append(color_col)
        plot_df = df[needed_cols].copy()
    else:
        plot_df = df.copy()

    # Ensure date column is datetime
    try:
        plot_df[date_col] = pd.to_datetime(plot_df[date_col])
    except:
        # If conversion fails, return a message
        fig = go.Figure()
        fig.add_annotation(
            text="Could not convert date column to datetime format",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Filter by date range if provided
    if date_range is not None and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date is not None:
            plot_df = plot_df[plot_df[date_col] >= pd.to_datetime(start_date)]
        if end_date is not None:
            plot_df = plot_df[plot_df[date_col] <= pd.to_datetime(end_date)]

    # Filter by selected categories if provided
    if color_col is not None and selected_categories is not None and len(selected_categories) > 0:
        plot_df = plot_df[plot_df[color_col].isin(selected_categories)]

    # Sort by date
    plot_df = plot_df.sort_values(by=date_col)

    # Downsample if the dataset is still too large after filtering
    if len(plot_df) > MAX_POINTS:
        if color_col is not None:
            # Downsample each category separately
            downsampled_dfs = []
            for category in plot_df[color_col].unique():
                category_df = plot_df[plot_df[color_col] == category]
                # Calculate points per category to maintain proportions
                category_max_points = max(100, int(MAX_POINTS * len(category_df) / len(plot_df)))
                downsampled_category = downsample_time_series(
                    category_df,
                    date_col,
                    value_col,
                    max_points=category_max_points
                )
                downsampled_dfs.append(downsampled_category)
            plot_df = pd.concat(downsampled_dfs)
        else:
            # Downsample the entire dataset
            plot_df = downsample_time_series(plot_df, date_col, value_col, max_points=MAX_POINTS)

    # Remove outliers if requested
    if remove_outliers:
        z_scores = np.abs((plot_df[value_col] - plot_df[value_col].mean()) / plot_df[value_col].std())
        plot_df = plot_df[z_scores < outlier_threshold]

    # Apply time aggregation if requested
    if aggregation != 'none':
        # Group by the specified time period
        if aggregation == 'day':
            plot_df[date_col] = plot_df[date_col].dt.date
        elif aggregation == 'week':
            plot_df[date_col] = plot_df[date_col].dt.to_period('W').dt.start_time
        elif aggregation == 'month':
            plot_df[date_col] = plot_df[date_col].dt.to_period('M').dt.start_time
        elif aggregation == 'quarter':
            plot_df[date_col] = plot_df[date_col].dt.to_period('Q').dt.start_time
        elif aggregation == 'year':
            plot_df[date_col] = plot_df[date_col].dt.to_period('Y').dt.start_time

        # If we have a color column, we need to group by both date and color
        if color_col:
            # Create a temporary date column for grouping
            plot_df['temp_date'] = plot_df[date_col]
            grouped = plot_df.groupby([plot_df['temp_date'], color_col])[value_col].agg(['mean', 'count']).reset_index()
            grouped.columns = [date_col, color_col, value_col, 'count']
        else:
            # Group by date only
            grouped = plot_df.groupby(plot_df[date_col])[value_col].agg(['mean', 'count']).reset_index()
            grouped.columns = [date_col, value_col, 'count']

        # Replace the original dataframe with the grouped one
        plot_df = grouped

        # Use the mean as the value
        if value_col in plot_df.columns:
            pass  # Already renamed in the color_col case
        else:
            plot_df[value_col] = plot_df['mean']

    # Apply rolling average if requested
    if rolling_window is not None and rolling_window > 1:
        if color_col:
            # Apply rolling average separately for each color group
            for color in plot_df[color_col].unique():
                mask = plot_df[color_col] == color
                plot_df.loc[mask, f'{value_col}_smooth'] = plot_df.loc[mask, value_col].rolling(window=rolling_window, min_periods=1).mean()

            # Use the smoothed values for plotting
            value_col_plot = f'{value_col}_smooth'
        else:
            # Apply rolling average to the entire series
            plot_df[f'{value_col}_smooth'] = plot_df[value_col].rolling(window=rolling_window, min_periods=1).mean()

            # Use the smoothed values for plotting
            value_col_plot = f'{value_col}_smooth'
    else:
        # Use the original values
        value_col_plot = value_col

    # Create the plot
    title_parts = []
    if aggregation != 'none':
        title_parts.append(f"{aggregation.capitalize()} Aggregated")
    if rolling_window is not None and rolling_window > 1:
        title_parts.append(f"{rolling_window}-Point Rolling Average")
    if remove_outliers:
        title_parts.append(f"Outliers Removed (z-score > {outlier_threshold})")

    title_prefix = " ".join(title_parts)
    title = f"{title_prefix + ' ' if title_prefix else ''}Trend of {value_col} over {date_col}"

    # Determine if we should use markers based on data size
    use_markers = len(plot_df) < 100

    # For large datasets, use more efficient Graph Objects instead of Express
    if len(plot_df) > 1000:
        fig = go.Figure()

        if color_col:
            # Create a trace for each category
            for category in plot_df[color_col].unique():
                category_df = plot_df[plot_df[color_col] == category]
                fig.add_trace(go.Scatter(
                    x=category_df[date_col],
                    y=category_df[value_col_plot],
                    mode='lines' if not use_markers else 'lines+markers',
                    name=str(category),
                    line=dict(width=2),
                    marker=dict(size=4) if use_markers else None
                ))
        else:
            # Create a single trace
            fig.add_trace(go.Scatter(
                x=plot_df[date_col],
                y=plot_df[value_col_plot],
                mode='lines' if not use_markers else 'lines+markers',
                name=value_col,
                line=dict(width=2),
                marker=dict(size=4) if use_markers else None
            ))
    else:
        # For smaller datasets, use Plotly Express for simplicity
        if color_col:
            fig = px.line(
                plot_df,
                x=date_col,
                y=value_col_plot,
                color=color_col,
                title=title,
                markers=use_markers
            )
        else:
            fig = px.line(
                plot_df,
                x=date_col,
                y=value_col_plot,
                title=title,
                markers=use_markers
            )

    # Add value labels at key points (min, max, last point)
    # Only add for datasets that aren't too large to avoid clutter
    if len(plot_df) < 500:
        # Find key points
        if color_col:
            # Handle each category separately
            for category in plot_df[color_col].unique():
                category_df = plot_df[plot_df[color_col] == category]

                # Get key points
                max_idx = category_df[value_col_plot].idxmax()
                min_idx = category_df[value_col_plot].idxmin()
                last_idx = category_df.index[-1]

                # Create unique indices list (in case min/max/last are the same point)
                key_indices = list(set([max_idx, min_idx, last_idx]))
                key_points = category_df.loc[key_indices]

                # Add labels
                fig.add_trace(go.Scatter(
                    x=key_points[date_col],
                    y=key_points[value_col_plot],
                    mode='text',
                    text=[f"{y:.2f}" for y in key_points[value_col_plot]],
                    textposition='top center',
                    textfont=dict(size=10),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        else:
            # Get key points
            max_idx = plot_df[value_col_plot].idxmax()
            min_idx = plot_df[value_col_plot].idxmin()
            last_idx = plot_df.index[-1]

            # Create unique indices list (in case min/max/last are the same point)
            key_indices = list(set([max_idx, min_idx, last_idx]))
            key_points = plot_df.loc[key_indices]

            # Add labels
            fig.add_trace(go.Scatter(
                x=key_points[date_col],
                y=key_points[value_col_plot],
                mode='text',
                text=[f"{y:.2f}" for y in key_points[value_col_plot]],
                textposition='top center',
                textfont=dict(size=10),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add a trace for the original data if we're showing a smoothed version
    # Only do this for smaller datasets to avoid performance issues
    if rolling_window is not None and rolling_window > 1 and len(plot_df) < 1000:
        if color_col:
            # Add original data as scatter points with low opacity
            for color in plot_df[color_col].unique():
                color_df = plot_df[plot_df[color_col] == color]
                fig.add_trace(go.Scatter(
                    x=color_df[date_col],
                    y=color_df[value_col],
                    mode='markers',
                    marker=dict(opacity=0.3, size=3),
                    name=f"{color} (Original)",
                    showlegend=False
                ))
        else:
            # Add original data as scatter points with low opacity
            fig.add_trace(go.Scatter(
                x=plot_df[date_col],
                y=plot_df[value_col],
                mode='markers',
                marker=dict(opacity=0.3, size=3),
                name="Original Data",
                showlegend=False
            ))

    # Improve layout with optimized settings and increased top margin to avoid overlapping
    fig.update_layout(
        title={
            'text': title,
            'y': 0.95,  # Position the title higher
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=date_col,
        yaxis_title=value_col,
        margin=dict(l=0, r=0, t=70, b=0),  # Increased top margin to accommodate range selector
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        uirevision='constant',  # Preserve UI state on updates
        # Optimize for performance
        autosize=True,
        height=500,
        template="plotly_white"  # Use a simpler template for better performance
    )

    # Add range slider for time navigation - only for datasets that aren't too large
    if len(plot_df) < 10000:  # Skip for very large datasets
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                # Position the range selector to avoid overlap with the title
                y=1.0,
                x=0.0,
                yanchor="top",
                xanchor="left"
            )
        )
    else:
        # For very large datasets, use a simpler x-axis configuration
        fig.update_xaxes(
            rangeslider_visible=False,
            showspikes=True,  # Show spikes for better navigation
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikedash="solid"
        )

    return fig
