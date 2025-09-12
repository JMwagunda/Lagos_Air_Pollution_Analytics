"""
Pollution trends analysis module for Lagos Air Pollution Analysis Project.
Performs exploratory data analysis focused on pollution trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def perform_pollution_eda(df):
    """Comprehensive Exploratory Data Analysis for pollution trends"""
    print("\n🚀 PERFORMING POLLUTION TRENDS EDA...")
    print("=" * 50)
    
    # Dataset overview
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    if 'city' in df.columns:
        print(f"Cities covered: {df['city'].nunique()}")
        print(f"Cities: {df['city'].unique()}")
    
    # Statistical summary for pollution metrics
    print("\n2. POLLUTION METRICS SUMMARY")
    print("-" * 30)
    
    pollution_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3', 'pollution_index']
    existing_pollution_cols = [col for col in pollution_cols if col in df.columns]
    
    if existing_pollution_cols:
        print(df[existing_pollution_cols].describe())
    
    # Missing values summary
    print("\n3. DATA QUALITY CHECK")
    print("-" * 30)
    
    missing_summary = df.isnull().sum()
    if missing_summary.sum() > 0:
        print("Missing values:")
        print(missing_summary[missing_summary > 0])
    else:
        print("No missing values detected")
    
    return df

def create_pollution_visualizations(df):
    """Create pollution-focused visualizations"""
    print("\n🚀 CREATING POLLUTION VISUALIZATIONS...")
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Pollution trends over time
    if 'date' in df.columns and 'pm2_5' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Air Pollution Analysis - Lagos', fontsize=16, fontweight='bold')
        
        # PM2.5 trends by city
        if 'city' in df.columns:
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                axes[0, 0].plot(city_data['date'], city_data['pm2_5'], 
                                label=city, marker='o', markersize=2)
        else:
            axes[0, 0].plot(df['date'], df['pm2_5'], marker='o', markersize=2)
        
        axes[0, 0].set_title('PM2.5 Levels Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('PM2.5 (µg/m³)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pollution index distribution
        if 'pollution_index' in df.columns:
            axes[0, 1].hist(df['pollution_index'], bins=30, 
                           color='lightcoral', edgecolor='darkred', alpha=0.7)
            axes[0, 1].set_title('Pollution Index Distribution')
            axes[0, 1].set_xlabel('Pollution Index')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly patterns
        if 'month' in df.columns:
            monthly_pollution = df.groupby('month')['pm2_5'].mean()
            axes[1, 0].bar(monthly_pollution.index, monthly_pollution.values,
                          color='skyblue', edgecolor='navy')
            axes[1, 0].set_title('Average PM2.5 by Month')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Average PM2.5 (µg/m³)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Seasonal analysis
        if 'season' in df.columns and 'pm2_5' in df.columns:
            sns.boxplot(data=df, x='season', y='pm2_5', palette='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('PM2.5 Levels by Season')
            axes[1, 1].set_ylabel('PM2.5 (µg/m³)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_trends_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Correlation heatmap for pollution metrics
    print("\n🚀 Creating Pollution Correlation Matrix...")
    
    # Define pollution features for the correlation matrix
    pollution_features = [
        'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'pollution_index',
    ]
    
    # Filter for existing pollution features in the DataFrame
    existing_pollution_features = [col for col in pollution_features if col in df.columns]
    
    if len(existing_pollution_features) > 1:  # Need at least 2 columns for correlation
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[existing_pollution_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, fmt=".2f")
        plt.title('Correlation Matrix of Pollution Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Pollution correlation matrix visualized.")
    else:
        print("Skipping pollution correlation matrix: Not enough pollution features found in the DataFrame.")
    
    # 3. Average Monthly Pollution Trends (All Cities Combined) with Pollutant Legends
    print("\n🚀 Visualizing Average Monthly Pollution Trends (All Cities Combined)...")
    
    # Select key pollutant columns
    pollutant_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
    existing_pollutant_cols = [col for col in pollutant_cols if col in df.columns]
    
    if 'date' in df.columns and existing_pollutant_cols:
        # Prepare data for plotting - melt the DataFrame to long format
        # Group by date and take the mean across all locations for each pollutant
        daily_avg_pollutants = df.groupby('date')[existing_pollutant_cols].mean().reset_index()
        
        # Melt the DataFrame to have pollutant names as a variable for seaborn hue
        pollutants_long = daily_avg_pollutants.melt(
            id_vars='date',
            value_vars=existing_pollutant_cols,
            var_name='Pollutant',
            value_name='Concentration'
        )
        
        # Resample to monthly average for smoother trends
        pollutants_long['month'] = pollutants_long['date'].dt.to_period('M')
        monthly_avg_pollutants = pollutants_long.groupby(['month', 'Pollutant'])['Concentration'].mean().reset_index()
        monthly_avg_pollutants['month'] = monthly_avg_pollutants['month'].dt.to_timestamp()
        
        plt.figure(figsize=(14, 7))
        sns.lineplot(data=monthly_avg_pollutants, x='month', y='Concentration', hue='Pollutant', marker='o')
        plt.title('Average Monthly Pollutant Concentrations Across All Cities', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Average Concentration (µg/m³)')
        plt.grid(True, alpha=0.6)
        plt.legend(title='Pollutant')
        plt.tight_layout()
        plt.savefig('outputs/figures/monthly_pollution_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Average Monthly Pollutant Trends visualized.")
    else:
        print("Skipping pollutant trend visualization: Required columns ('date' and at least one pollutant) not found.")

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
    
    # Perform pollution EDA
    df = perform_pollution_eda(df)
    
    # Create pollution visualizations
    create_pollution_visualizations(df)
    
    print("🚀 Pollution trends analysis complete.")

if __name__ == "__main__":
    main()