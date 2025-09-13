"""
Dashboard creation module for Lagos Air Pollution Analysis Project.
Creates an interactive dashboard to visualize the analysis results.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

def create_dashboard(df):
    """Create an interactive dashboard for the Lagos Air Pollution Analysis"""
    print("\n🚀 CREATING INTERACTIVE DASHBOARD")
    print("=" * 50)
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('PM2.5 Levels Over Time', 'Respiratory Cases Over Time',
                       'Pollution vs Health Correlation', 'City-wise Pollution Comparison',
                       'Seasonal Pollution Patterns', 'Pollution Index Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. PM2.5 Levels Over Time
    if 'date' in df.columns and 'pm2_5' in df.columns:
        # Group by date and calculate mean PM2.5
        daily_pm25 = df.groupby('date')['pm2_5'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(x=daily_pm25['date'], y=daily_pm25['pm2_5'],
                      mode='lines', name='PM2.5',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
    
    # 2. Respiratory Cases Over Time
    if 'date' in df.columns and 'respiratory_cases' in df.columns:
        # Group by date and calculate mean respiratory cases
        daily_cases = df.groupby('date')['respiratory_cases'].mean().reset_index()
        
        fig.add_trace(
            go.Scatter(x=daily_cases['date'], y=daily_cases['respiratory_cases'],
                      mode='lines', name='Respiratory Cases',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
    
    # 3. Pollution vs Health Correlation
    if 'pm2_5' in df.columns and 'respiratory_cases' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['pm2_5'], y=df['respiratory_cases'],
                      mode='markers', name='Data Points',
                      marker=dict(color='green', size=5, opacity=0.5)),
            row=2, col=1
        )
        
        # Add trend line
        z = np.polyfit(df['pm2_5'], df['respiratory_cases'], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(x=df['pm2_5'], y=p(df['pm2_5']),
                      mode='lines', name='Trend Line',
                      line=dict(color='red', width=2, dash='dash')),
            row=2, col=1
        )
    
    # 4. City-wise Pollution Comparison
    if 'city' in df.columns and 'pm2_5' in df.columns:
        city_pm25 = df.groupby('city')['pm2_5'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(x=city_pm25['city'], y=city_pm25['pm2_5'],
                   name='City PM2.5',
                   marker_color='skyblue'),
            row=2, col=2
        )
    
    # 5. Seasonal Pollution Patterns
    if 'season' in df.columns and 'pm2_5' in df.columns:
        season_pm25 = df.groupby('season')['pm2_5'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(x=season_pm25['season'], y=season_pm25['pm2_5'],
                   name='Seasonal PM2.5',
                   marker_color='lightgreen'),
            row=3, col=1
        )
    
    # 6. Pollution Index Distribution
    if 'pollution_index' in df.columns:
        fig.add_trace(
            go.Histogram(x=df['pollution_index'],
                       name='Pollution Index',
                       marker_color='salmon'),
            row=3, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Lagos Air Pollution and Health Dashboard",
        title_font_size=20,
        showlegend=False,
        height=900,
        width=1200
    )
    
    # Update x and y axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", row=1, col=1)
    
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Respiratory Cases", row=1, col=2)
    
    fig.update_xaxes(title_text="PM2.5 (µg/m³)", row=2, col=1)
    fig.update_yaxes(title_text="Respiratory Cases", row=2, col=1)
    
    fig.update_xaxes(title_text="City", row=2, col=2)
    fig.update_yaxes(title_text="Average PM2.5 (µg/m³)", row=2, col=2)
    
    fig.update_xaxes(title_text="Season", row=3, col=1)
    fig.update_yaxes(title_text="Average PM2.5 (µg/m³)", row=3, col=1)
    
    fig.update_xaxes(title_text="Pollution Index", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)
    
    # Save dashboard
    dashboard_path = 'outputs/figures/interactive_dashboard.html'
    try:
        fig.write_html(dashboard_path)
        print(f"🚀 Interactive dashboard saved to {dashboard_path}")
    except Exception as e:
        print(f"🚀 Error saving dashboard: {e}")
    
    # Show dashboard
    fig.show()
    
    return fig

def create_city_comparison_dashboard(df):
    """Create a city-specific comparison dashboard"""
    print("\n🚀 CREATING CITY COMPARISON DASHBOARD")
    print("=" * 50)
    
    # Check if required columns exist
    required_cols = ['city', 'pm2_5', 'respiratory_cases']
    if not all(col in df.columns for col in required_cols):
        print("Skipping city comparison dashboard: Required columns not found.")
        return None
    
    # Get unique cities
    cities = df['city'].unique()
    
    # Create a figure with subplots for each city
    fig = make_subplots(
        rows=len(cities), cols=1,
        subplot_titles=[f"{city} - PM2.5 vs Respiratory Cases" for city in cities]
    )
    
    # Add traces for each city
    for i, city in enumerate(cities):
        city_data = df[df['city'] == city]
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(x=city_data['pm2_5'], y=city_data['respiratory_cases'],
                      mode='markers', name=city,
                      marker=dict(size=6, opacity=0.7)),
            row=i+1, col=1
        )
        
        # Add trend line
        if len(city_data) > 1:  # Need at least 2 points for a trend line
            z = np.polyfit(city_data['pm2_5'], city_data['respiratory_cases'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(x=city_data['pm2_5'], y=p(city_data['pm2_5']),
                          mode='lines', name=f"{city} Trend",
                          line=dict(dash='dash')),
                row=i+1, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="City-wise Pollution vs Health Impact",
        title_font_size=20,
        height=300 * len(cities),
        width=800,
        showlegend=False
    )
    
    # Update x and y axes labels
    for i in range(len(cities)):
        fig.update_xaxes(title_text="PM2.5 (µg/m³)", row=i+1, col=1)
        fig.update_yaxes(title_text="Respiratory Cases", row=i+1, col=1)
    
    # Save dashboard
    dashboard_path = 'outputs/figures/city_comparison_dashboard.html'
    try:
        fig.write_html(dashboard_path)
        print(f"🚀 City comparison dashboard saved to {dashboard_path}")
    except Exception as e:
        print(f"🚀 Error saving city comparison dashboard: {e}")
    
    # Show dashboard
    fig.show()
    
    return fig

def create_time_series_dashboard(df):
    """Create a time series dashboard for pollution and health metrics"""
    print("\n🚀 CREATING TIME SERIES DASHBOARD")
    print("=" * 50)
    
    # Check if required columns exist
    required_cols = ['date', 'pm2_5', 'respiratory_cases']
    if not all(col in df.columns for col in required_cols):
        print("Skipping time series dashboard: Required columns not found.")
        return None
    
    # Create a figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Monthly Average Pollution Metrics', 'Monthly Average Health Metrics'),
        specs=[[{"secondary_y": True}],
               [{"secondary_y": True}]]
    )
    
    # Prepare data - group by month
    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_data = df.groupby('month').mean(numeric_only=True).reset_index()
    
    # Convert month back to datetime for plotting
    monthly_data['month'] = pd.to_datetime(monthly_data['month'])
    
    # 1. Monthly Average Pollution Metrics
    pollution_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, col in enumerate(pollution_cols):
        if col in monthly_data.columns:
            fig.add_trace(
                go.Scatter(x=monthly_data['month'], y=monthly_data[col],
                          mode='lines+markers', name=col.upper(),
                          line=dict(color=colors[i % len(colors)], width=2)),
                row=1, col=1
            )
    
    # 2. Monthly Average Health Metrics
    health_cols = ['respiratory_cases', 'avg_age_of_patients']
    
    for i, col in enumerate(health_cols):
        if col in monthly_data.columns:
            fig.add_trace(
                go.Scatter(x=monthly_data['month'], y=monthly_data[col],
                          mode='lines+markers', name=col.replace('_', ' ').title(),
                          line=dict(color=colors[i % len(colors)], width=2)),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Time Series Dashboard - Pollution and Health Metrics",
        title_font_size=20,
        height=800,
        width=1200
    )
    
    # Update x and y axes labels
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_yaxes(title_text="Pollution Concentration (µg/m³)", row=1, col=1)
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Health Metrics", row=2, col=1)
    
    # Save dashboard
    dashboard_path = 'outputs/figures/time_series_dashboard.html'
    try:
        fig.write_html(dashboard_path)
        print(f"🚀 Time series dashboard saved to {dashboard_path}")
    except Exception as e:
        print(f"🚀 Error saving time series dashboard: {e}")
    
    # Show dashboard
    fig.show()
    
    return fig

def main():
    # Define file paths
    engineered_data_path = "data/processed/pollution_index_data.parquet"
    
    # Load engineered data
    try:
        df = pd.read_parquet(engineered_data_path)
        print(f"🚀 Loaded engineered data: {df.shape}")
    except Exception as e:
        print(f"🚀 Error loading engineered data: {e}")
        return
    
    # Create main dashboard
    main_dashboard = create_dashboard(df)
    
    # Create city comparison dashboard
    city_dashboard = create_city_comparison_dashboard(df)
    
    # Create time series dashboard
    time_dashboard = create_time_series_dashboard(df)
    
    print("\n🚀 Dashboard creation complete.")

if __name__ == "__main__":
    main()