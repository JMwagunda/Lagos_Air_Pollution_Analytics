"""
Feature engineering module for Lagos Air Pollution Analysis Project.
Creates derived features from the cleaned dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def create_derived_features(df):
    """Create new features from existing data"""
    print("\n⚙️ CREATING DERIVED FEATURES...")
    
    # Create pollution index (composite feature)
    pollution_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
    existing_pollution_cols = [col for col in pollution_cols if col in df.columns]
    
    if len(existing_pollution_cols) >= 3:
        print(f"-> Found pollution columns: {existing_pollution_cols}")
        
        # Normalize each pollutant to 0-100 scale before combining
        scaler = StandardScaler()
        normalized_pollution = scaler.fit_transform(df[existing_pollution_cols])
        
        # Create weighted pollution index (PM2.5 and PM10 have higher weights)
        weights = {'pm2_5': 0.3, 'pm10': 0.25, 'no2': 0.2, 'so2': 0.15, 'o3': 0.1}
        df['pollution_index'] = 0
        
        for i, col in enumerate(existing_pollution_cols):
            weight = weights.get(col, 0.1)
            df['pollution_index'] += normalized_pollution[:, i] * weight
        
        # Scale to 0-100
        df['pollution_index'] = (df['pollution_index'] - df['pollution_index'].min()) / \
                                (df['pollution_index'].max() - df['pollution_index'].min()) * 100
        
        print("-> Created 'pollution_index'.")
        print(df['pollution_index'].head())
    
    # Create time-based features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['quarter'] = df['date'].dt.quarter
        df['is_harmattan'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        
        df['season'] = df['month'].map({
            12: 'Dry', 1: 'Dry', 2: 'Dry',
            3: 'Dry', 4: 'Wet', 5: 'Wet',
            6: 'Wet', 7: 'Wet', 8: 'Wet',
            9: 'Wet', 10: 'Dry', 11: 'Dry'
        })
        
        print("-> Created time-based features ('year', 'month', 'day_of_year', 'quarter', 'is_harmattan', 'season').")
    
    # Create air quality categories
    if 'pm2_5' in df.columns:
        df['pm2_5_category'] = pd.cut(df['pm2_5'],
                                     bins=[0, 15, 30, 55, 110, float('inf')],
                                     labels=['Good', 'Moderate', 'Unhealthy for Sensitive', 
                                            'Unhealthy', 'Very Unhealthy'])
        print("-> Created 'pm2_5_category'.")
    
    # Create health risk indicators
    if 'respiratory_cases' in df.columns:
        df['high_respiratory_risk'] = (df['respiratory_cases'] > 
                                      df['respiratory_cases'].quantile(0.75)).astype(int)
        print("-> Created 'high_respiratory_risk'.")
    
    # Create population-adjusted metrics
    if 'respiratory_cases' in df.columns and 'population_density' in df.columns:
        df['cases_per_thousand'] = (df['respiratory_cases'] / df['population_density']) * 1000
        print("-> Created 'cases_per_thousand'.")
    
    # Create weather comfort index
    weather_cols = ['weather_temperature', 'weather_humidity']
    if all(col in df.columns for col in weather_cols):
        # Simple comfort index based on temperature and humidity
        df['weather_comfort_index'] = (
            np.where(df['weather_temperature'].between(20, 30) & 
                    df['weather_humidity'].between(40, 60), 1, 0)
        )
        print("-> Created 'weather_comfort_index'.")
    
    new_features = [
        'pollution_index', 'year', 'month', 'season', 'is_harmattan',
        'pm2_5_category', 'high_respiratory_risk',
        'cases_per_thousand', 'weather_comfort_index'
    ]
    
    created = [col for col in df.columns if col in new_features]
    print(f"\n🚀 Created {len(created)} new features: {created}")
    print("🚀 Feature creation complete.")
    
    return df

def main():
    # Define file paths
    cleaned_data_path = "data/processed/cleaned_data.parquet"
    engineered_data_path = "data/processed/pollution_index_data.parquet"
    
    # Load cleaned data
    try:
        df = pd.read_parquet(cleaned_data_path)
        print(f"🚀 Loaded cleaned data: {df.shape}")
    except Exception as e:
        print(f"🚀 Error loading cleaned data: {e}")
        return
    
    # Create derived features
    df_engineered = create_derived_features(df)
    
    # Save engineered dataset
    try:
        df_engineered.to_parquet(engineered_data_path, index=False)
        print(f"🚀 Engineered dataset saved as '{engineered_data_path}'")
    except Exception as e:
        print(f"🚀 Error saving engineered dataset: {e}")

if __name__ == "__main__":
    main()