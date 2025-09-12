"""
Spatial analysis module for Lagos Air Pollution Analysis Project.
Performs exploratory data analysis focused on spatial patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def perform_spatial_eda(df):
    """Comprehensive Exploratory Data Analysis for spatial patterns"""
    print("\n🚀 PERFORMING SPATIAL ANALYSIS EDA...")
    print("=" * 50)
    
    # Dataset overview
    print("\n1. SPATIAL DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    
    if 'city' in df.columns:
        print(f"Cities covered: {df['city'].nunique()}")
        print(f"Cities: {df['city'].unique()}")
    
    # Statistical summary by city
    print("\n2. CITY-WISE STATISTICS")
    print("-" * 30)
    
    if 'city' in df.columns:
        # Group by city and calculate mean values
        city_stats = df.groupby('city').mean(numeric_only=True)
        
        # Select key metrics for display
        key_metrics = ['pm2_5', 'pm10', 'respiratory_cases', 'population_density', 'industrial_activity_index']
        existing_metrics = [col for col in key_metrics if col in city_stats.columns]
        
        if existing_metrics:
            print(city_stats[existing_metrics])
    
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

def create_spatial_visualizations(df):
    """Create spatial-focused visualizations"""
    print("\n🚀 CREATING SPATIAL VISUALIZATIONS...")
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. City-wise pollution comparison
    if 'city' in df.columns and 'pm2_5' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Analysis - Lagos', fontsize=16, fontweight='bold')
        
        # PM2.5 by city
        city_pm25 = df.groupby('city')['pm2_5'].mean().sort_values(ascending=False)
        axes[0, 0].bar(city_pm25.index, city_pm25.values, color='skyblue', edgecolor='navy')
        axes[0, 0].set_title('Average PM2.5 by City')
        axes[0, 0].set_ylabel('PM2.5 (µg/m³)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Respiratory cases by city
        if 'respiratory_cases' in df.columns:
            city_cases = df.groupby('city')['respiratory_cases'].mean().sort_values(ascending=False)
            axes[0, 1].bar(city_cases.index, city_cases.values, color='salmon', edgecolor='darkred')
            axes[0, 1].set_title('Average Respiratory Cases by City')
            axes[0, 1].set_ylabel('Respiratory Cases')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Population density by city
        if 'population_density' in df.columns:
            city_pop = df.groupby('city')['population_density'].mean().sort_values(ascending=False)
            axes[1, 0].bar(city_pop.index, city_pop.values, color='lightgreen', edgecolor='darkgreen')
            axes[1, 0].set_title('Average Population Density by City')
            axes[1, 0].set_ylabel('Population Density')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Industrial activity by city
        if 'industrial_activity_index' in df.columns:
            city_industry = df.groupby('city')['industrial_activity_index'].mean().sort_values(ascending=False)
            axes[1, 1].bar(city_industry.index, city_industry.values, color='purple', edgecolor='darkviolet')
            axes[1, 1].set_title('Average Industrial Activity by City')
            axes[1, 1].set_ylabel('Industrial Activity Index')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/spatial_city_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Pollution vs Population Density
    if 'city' in df.columns and 'pm2_5' in df.columns and 'population_density' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Get city-wise averages
        city_data = df.groupby('city')[['pm2_5', 'population_density']].mean().reset_index()
        
        # Create scatter plot with city labels
        plt.scatter(city_data['population_density'], city_data['pm2_5'], s=100, alpha=0.7)
        
        # Add city labels
        for i, row in city_data.iterrows():
            plt.text(row['population_density'], row['pm2_5'], row['city'], 
                    fontsize=12, ha='right', va='bottom')
        
        # Add trend line
        z = np.polyfit(city_data['population_density'], city_data['pm2_5'], 1)
        p = np.poly1d(z)
        plt.plot(city_data['population_density'], p(city_data['population_density']), "r--", alpha=0.8)
        
        plt.title('PM2.5 vs Population Density by City', fontsize=14, fontweight='bold')
        plt.xlabel('Population Density')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_vs_population_density.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Pollution vs Population Density analysis visualized.")
    
    # 3. Industrial Activity vs Pollution
    if 'city' in df.columns and 'pm2_5' in df.columns and 'industrial_activity_index' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Get city-wise averages
        city_data = df.groupby('city')[['pm2_5', 'industrial_activity_index']].mean().reset_index()
        
        # Create scatter plot with city labels
        plt.scatter(city_data['industrial_activity_index'], city_data['pm2_5'], s=100, alpha=0.7)
        
        # Add city labels
        for i, row in city_data.iterrows():
            plt.text(row['industrial_activity_index'], row['pm2_5'], row['city'], 
                    fontsize=12, ha='right', va='bottom')
        
        # Add trend line
        z = np.polyfit(city_data['industrial_activity_index'], city_data['pm2_5'], 1)
        p = np.poly1d(z)
        plt.plot(city_data['industrial_activity_index'], p(city_data['industrial_activity_index']), "r--", alpha=0.8)
        
        plt.title('PM2.5 vs Industrial Activity by City', fontsize=14, fontweight='bold')
        plt.xlabel('Industrial Activity Index')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_vs_industrial_activity.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Pollution vs Industrial Activity analysis visualized.")
    
    # 4. City-wise seasonal patterns
    if 'city' in df.columns and 'season' in df.columns and 'pm2_5' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Create a pivot table for the heatmap
        season_city_pm25 = df.pivot_table(
            values='pm2_5', 
            index='city', 
            columns='season', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(season_city_pm25, annot=True, cmap='YlOrRd', fmt='.1f')
        plt.title('Average PM2.5 by City and Season', fontsize=14, fontweight='bold')
        plt.xlabel('Season')
        plt.ylabel('City')
        plt.tight_layout()
        plt.savefig('outputs/figures/city_seasonal_pm25_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 City-wise seasonal PM2.5 heatmap visualized.")

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
    
    # Perform spatial EDA
    df = perform_spatial_eda(df)
    
    # Create spatial visualizations
    create_spatial_visualizations(df)
    
    print("🚀 Spatial analysis complete.")

if __name__ == "__main__":
    main()