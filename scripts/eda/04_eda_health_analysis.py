"""
Health analysis module for Lagos Air Pollution Analysis Project.
Performs exploratory data analysis focused on health impacts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def perform_health_eda(df):
    """Comprehensive Exploratory Data Analysis for health impacts"""
    print("\n🚀 PERFORMING HEALTH IMPACTS EDA...")
    print("=" * 50)
    
    # Dataset overview
    print("\n1. HEALTH DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    if 'city' in df.columns:
        print(f"Cities covered: {df['city'].nunique()}")
        print(f"Cities: {df['city'].unique()}")
    
    # Statistical summary for health metrics
    print("\n2. HEALTH METRICS SUMMARY")
    print("-" * 30)
    
    health_cols = ['respiratory_cases', 'avg_age_of_patients', 'high_respiratory_risk', 'cases_per_thousand']
    existing_health_cols = [col for col in health_cols if col in df.columns]
    
    if existing_health_cols:
        print(df[existing_health_cols].describe())
    
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

def create_health_visualizations(df):
    """Create health-focused visualizations"""
    print("\n🚀 CREATING HEALTH VISUALIZATIONS...")
    
    # Set up the plotting parameters
    plt.rcParams['figure.figsize'] = (15, 10)
    
    # 1. Health trends over time
    if 'date' in df.columns and 'respiratory_cases' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Health Impact Analysis - Lagos', fontsize=16, fontweight='bold')
        
        # Respiratory cases trends by city
        if 'city' in df.columns:
            for city in df['city'].unique():
                city_data = df[df['city'] == city]
                axes[0, 0].plot(city_data['date'], city_data['respiratory_cases'], 
                                label=city, marker='o', markersize=2)
        else:
            axes[0, 0].plot(df['date'], df['respiratory_cases'], marker='o', markersize=2)
        
        axes[0, 0].set_title('Respiratory Cases Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Respiratory Cases')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Age distribution of patients
        if 'avg_age_of_patients' in df.columns:
            axes[0, 1].hist(df['avg_age_of_patients'], bins=30, 
                           color='lightgreen', edgecolor='darkgreen', alpha=0.7)
            axes[0, 1].set_title('Age Distribution of Patients')
            axes[0, 1].set_xlabel('Average Age')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly respiratory cases patterns
        if 'month' in df.columns:
            monthly_cases = df.groupby('month')['respiratory_cases'].mean()
            axes[1, 0].bar(monthly_cases.index, monthly_cases.values,
                          color='salmon', edgecolor='darkred')
            axes[1, 0].set_title('Average Respiratory Cases by Month')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Average Respiratory Cases')
            axes[1, 0].grid(True, alpha=0.3)
        
        # High respiratory risk by season
        if 'season' in df.columns and 'high_respiratory_risk' in df.columns:
            sns.boxplot(data=df, x='season', y='respiratory_cases', palette='viridis', ax=axes[1, 1])
            axes[1, 1].set_title('Respiratory Cases by Season')
            axes[1, 1].set_ylabel('Respiratory Cases')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/health_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Correlation heatmap for health metrics
    print("\n🚀 Creating Health Correlation Matrix...")
    
    # Define health features for the correlation matrix
    health_features = [
        'respiratory_cases', 'avg_age_of_patients', 'high_respiratory_risk', 
        'cases_per_thousand', 'pm2_5', 'pm10'
    ]
    
    # Filter for existing health features in the DataFrame
    existing_health_features = [col for col in health_features if col in df.columns]
    
    if len(existing_health_features) > 1:  # Need at least 2 columns for correlation
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[existing_health_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, fmt=".2f")
        plt.title('Correlation Matrix of Health Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/figures/health_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Health correlation matrix visualized.")
    else:
        print("Skipping health correlation matrix: Not enough health features found in the DataFrame.")
    
    # 3. Pollution vs Health Impact Analysis
    print("\n🚀 Analyzing Pollution vs Health Impact...")
    
    if 'pm2_5' in df.columns and 'respiratory_cases' in df.columns:
        plt.figure(figsize=(12, 8))
        plt.scatter(df['pm2_5'], df['respiratory_cases'], alpha=0.6, c='red')
        plt.title('PM2.5 vs Respiratory Cases', fontsize=14, fontweight='bold')
        plt.xlabel('PM2.5 (µg/m³)')
        plt.ylabel('Respiratory Cases')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(df['pm2_5'], df['respiratory_cases'], 1)
        p = np.poly1d(z)
        plt.plot(df['pm2_5'], p(df['pm2_5']), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_vs_health.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Pollution vs Health Impact analysis visualized.")
    else:
        print("Skipping Pollution vs Health analysis: Required columns not found.")
    
    # 4. Overlay Plot: Pollution Index vs Respiratory Cases (Monthly Average)
    print("\n🚀 Creating Overlay Plot: Pollution Index vs Respiratory Cases...")
    
    if 'date' in df.columns and 'pollution_index' in df.columns and 'respiratory_cases' in df.columns:
        # Calculate monthly average pollution index
        monthly_pollution_index = df.set_index('date')['pollution_index'].resample('M').mean()
        
        # Calculate monthly average respiratory cases
        monthly_cases = df.set_index('date')['respiratory_cases'].resample('M').mean()
        
        # Create a figure and a primary axes
        fig, ax1 = plt.subplots(figsize=(14, 7))
        
        # Plot Average Pollution Index on the primary axes
        ax1.plot(monthly_pollution_index.index, monthly_pollution_index.values, 
                color='tab:blue', label='Average Pollution Index')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Average Pollution Index', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, alpha=0.6)
        
        # Create a secondary axes that shares the same x-axis
        ax2 = ax1.twinx()
        
        # Plot Average Respiratory Cases on the secondary axes
        ax2.plot(monthly_cases.index, monthly_cases.values, 
                color='tab:red', label='Average Respiratory Cases')
        ax2.set_ylabel('Average Respiratory Cases', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Add a title
        plt.title('Monthly Average Pollution Index and Respiratory Cases Over Time', 
                 fontsize=14, fontweight='bold')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('outputs/figures/pollution_health_overlay.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("🚀 Overlay plot for Pollution Index and Respiratory Cases created.")
    else:
        print("Skipping overlay plot: Required columns ('date', 'pollution_index', 'respiratory_cases') not found.")

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
    
    # Perform health EDA
    df = perform_health_eda(df)
    
    # Create health visualizations
    create_health_visualizations(df)
    
    print("🚀 Health impact analysis complete.")

if __name__ == "__main__":
    main()