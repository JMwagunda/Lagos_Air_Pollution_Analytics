"""
Utility functions for the Streamlit dashboard.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_correlation_heatmap(df, columns):
    """Create a correlation heatmap for selected columns"""
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=500,
        width=600
    )
    
    return fig

def create_time_series_comparison(df, metrics, title="Time Series Comparison"):
    """Create a time series chart comparing multiple metrics"""
    # Group by date and calculate mean
    daily_data = df.groupby('date')[metrics].mean().reset_index()
    
    # Melt data for plotting
    daily_data_melted = daily_data.melt(id_vars='date', value_vars=metrics, 
                                        var_name='Metric', value_name='Value')
    
    # Create line chart
    fig = px.line(
        daily_data_melted,
        x='date',
        y='Value',
        color='Metric',
        title=title,
        labels={'Value': 'Value', 'date': 'Date'}
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_distribution_chart(df, column, title="Distribution"):
    """Create a distribution chart for a column"""
    fig = px.histogram(
        df,
        x=column,
        title=title,
        labels={column: column}
    )
    
    fig.update_layout(
        height=400
    )
    
    return fig

def create_box_plot(df, x_column, y_column, title="Box Plot"):
    """Create a box plot comparing categories"""
    fig = px.box(
        df,
        x=x_column,
        y=y_column,
        title=title,
        labels={x_column: x_column, y_column: y_column}
    )
    
    fig.update_layout(
        height=500
    )
    
    return fig

def get_summary_stats(df, column):
    """Get summary statistics for a column"""
    stats = {
        'Mean': df[column].mean(),
        'Median': df[column].median(),
        'Std Dev': df[column].std(),
        'Min': df[column].min(),
        'Max': df[column].max(),
        '25th Percentile': df[column].quantile(0.25),
        '75th Percentile': df[column].quantile(0.75)
    }
    
    return stats