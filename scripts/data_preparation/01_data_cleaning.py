"""
Data cleaning module for Lagos Air Pollution Analysis Project.
Handles loading, inspecting, and cleaning raw air pollution and health data.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

def load_and_inspect_data(file_path1, file_path2):
    """Load the datasets and perform initial inspection"""
    print("🚀 LOADING DATASETS...")
    
    # Load the datasets
    try:
        df1 = pd.read_csv(file_path1)
        df2 = pd.read_csv(file_path2)
        print(f"🚀 Dataset 1 loaded successfully: {df1.shape}")
        print(f"🚀 Dataset 2 loaded successfully: {df2.shape}")
    except FileNotFoundError as e:
        print(f"🚀 Error loading datasets: {e}")
        return None, None
    
    # Initial inspection
    print("\n🚀 INITIAL DATA INSPECTION")
    print("-" * 40)
    
    print("\nDataset 1 Info:")
    print(f"Shape: {df1.shape}")
    print(f"Columns: {list(df1.columns)}")
    print(f"Memory usage: {df1.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nDataset 2 Info:")
    print(f"Shape: {df2.shape}")
    print(f"Columns: {list(df2.columns)}")
    print(f"Memory usage: {df2.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    print("\nMissing Values in Dataset 1:")
    missing1 = df1.isnull().sum()
    print(missing1[missing1 > 0])
    
    print("\nMissing Values in Dataset 2:")
    missing2 = df2.isnull().sum()
    print(missing2[missing2 > 0])
    
    # Display sample data
    print("\nSample from Dataset 1:")
    print(df1.head(3))
    
    print("\nSample from Dataset 2:")
    print(df2.head(3))
    
    return df1, df2

def standardize_column_names(df):
    """Standardize column names across datasets"""
    print("-> Standardizing column names...")
    
    # Standardize all column names to lowercase and replace spaces
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Handle common variations
    column_mapping = {
        'c': 'city',
        'city': 'city'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    print("🚀 Column names standardized.")
    return df

def clean_city_names(df):
    """Clean and standardize city names"""
    print("-> Cleaning city names...")
    
    if 'city' in df.columns:
        city_mapping = {
            'AJAH': 'Ajah',
            'ajah': 'Ajah',
            'I K E J A': 'Ikeja',
            'IKEJA': 'Ikeja',
            'ikeja': 'Ikeja',
            'Ikeja': 'Ikeja',
            'YABA': 'Yaba',
            'yaba': 'Yaba',
            'SURULERE': 'Surulere',
            'surulere': 'Surulere',
            'LEKKI': 'Lekki',
            'lekki': 'Lekki',
            'A J A H': 'Ajah'
        }
        
        df['city'] = df['city'].replace(city_mapping)
        df['city'] = df['city'].str.strip()
        
        print("🚀 City names cleaned.")
    
    return df

def clean_and_convert_data_types(df):
    """Clean and convert data types appropriately"""
    print("\n🚀 CLEANING AND CONVERTING DATA TYPES...")
    
    # Clean date column - handle various date formats
    if 'date' in df.columns:
        # Attempt to parse with dayfirst=True first
        df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
        
        # If still NaT, try without dayfirst (assuming MM/DD/YYYY or similar)
        if df['date'].isnull().sum() > 0:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Handle remaining missing dates - impute with the mode
        if df['date'].isnull().sum() > 0:
            mode_date = df['date'].mode()[0] if not df['date'].mode().empty else pd.NaT
            if pd.notna(mode_date):
                df['date'] = df['date'].fillna(mode_date)
                print(f"-> Imputed {df['date'].isnull().sum()} missing dates with mode: {mode_date}.")
            else:
                print("-> Could not impute missing dates as mode is not available.")
        
        print("-> Date column cleaned and converted.")
    
    # Define numeric columns
    numeric_columns = [
        'pm2_5', 'pm10', 'no2', 'so2', 'o3', 'respiratory_cases',
        'avg_age_of_patients', 'weather_temperature', 'weather_humidity',
        'wind_speed', 'rainfall_mm', 'population_density',
        'industrial_activity_index'
    ]
    
    # Convert numeric columns
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("-> Numeric columns converted.")
    
    # Clean hospital_id
    if 'hospital_id' in df.columns:
        df['hospital_id'] = df['hospital_id'].astype(str).str.strip()
        print("-> Hospital ID cleaned.")
    
    print(f"🚀 Data types converted successfully")
    return df

def handle_missing_values(df):
    """Handle missing values using various imputation strategies"""
    print("\n🚀 HANDLING MISSING VALUES...")
    
    # Check missing values before imputation
    missing_before = df.isnull().sum()
    print(f"Total missing values before imputation: {missing_before.sum()}")
    
    if missing_before.sum() > 0:
        print("Missing values before imputation:")
        print(missing_before[missing_before > 0])
    
    # Strategy 1: Forward fill for time series data
    time_series_cols = ['pm2_5', 'pm10', 'no2', 'so2', 'o3']
    for col in time_series_cols:
        if col in df.columns:
            df[col] = df.groupby('city')[col].fillna(method='ffill')
            df[col] = df.groupby('city')[col].fillna(method='bfill')
    
    print("-> Time series columns imputed using ffill and bfill.")
    
    # Strategy 2: Mean imputation for weather data (grouped by city and month)
    weather_cols = ['weather_temperature', 'weather_humidity', 'wind_speed', 'rainfall_mm']
    for col in weather_cols:
        if col in df.columns and 'date' in df.columns:
            df['month'] = df['date'].dt.month
            df[col] = df.groupby(['city', 'month'])[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    print("-> Weather columns imputed using grouped mean.")
    
    # Strategy 3: KNN imputation for remaining numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
    
    print("-> Remaining numerical columns imputed using KNN.")
    
    # Strategy 4: Mode imputation for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col != 'date':
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_value)
    
    print("-> Categorical columns imputed using mode.")
    
    # Check missing values after imputation
    missing_after = df.isnull().sum()
    print(f"Total missing values after imputation: {missing_after.sum()}")
    
    if missing_after.sum() > 0:
        print("Missing values after imputation:")
        print(missing_after[missing_after > 0])
    else:
        print("🚀 No missing values remaining.")
    
    return df

def remove_duplicates(df, subset_cols=None):
    """Remove duplicate records with a more cautious approach."""
    print("\n REMOVING DUPLICATES...")
    
    initial_shape = df.shape
    print(f"Initial dataset shape: {initial_shape}")
    
    # Remove exact duplicates across all columns
    df_exact_duplicates_removed = df.drop_duplicates()
    exact_duplicates_removed_count = initial_shape[0] - df_exact_duplicates_removed.shape[0]
    print(f"Shape after removing exact duplicates: {df_exact_duplicates_removed.shape}")
    print(f"Removed {exact_duplicates_removed_count} exact duplicate records.")
    
    df = df_exact_duplicates_removed  # Continue with the dataframe after removing exact duplicates
    
    # Optional: Remove duplicates based on a subset of key columns
    if subset_cols:
        existing_subset_cols = [col for col in subset_cols if col in df.columns]
        if existing_subset_cols:
            print(f"\nAttempting to remove duplicates based on subset: {existing_subset_cols}")
            df_subset_duplicates_removed = df.drop_duplicates(subset=existing_subset_cols, keep='first')
            subset_duplicates_removed_count = df.shape[0] - df_subset_duplicates_removed.shape[0]
            df = df_subset_duplicates_removed
            print(f"Shape after removing subset duplicates: {df.shape}")
            print(f"Removed {subset_duplicates_removed_count} subset duplicate records.")
        else:
            print("Warning: None of the specified subset columns were found in the DataFrame. Skipping subset duplicate removal.")
    
    final_shape = df.shape
    total_removed_duplicates = initial_shape[0] - final_shape[0]
    print(f"\nTotal removed duplicate records: {total_removed_duplicates}")
    print(f"Final dataset shape: {final_shape}")
    print("🚀 Duplicate removal complete.")
    
    return df

def detect_and_handle_outliers(df):
    """Detect and handle outliers using IQR method"""
    print("\n🚀 DETECTING AND HANDLING OUTLIERS...")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_summary = {}
    
    for col in numerical_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                outlier_summary[col] = outlier_count
                # Cap outliers instead of removing them (to preserve data)
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    if outlier_summary:
        print("Outliers detected and capped:")
        for col, count in outlier_summary.items():
            print(f" {col}: {count} outliers")
        print("🚀 Outlier handling complete.")
    else:
        print("No significant outliers detected")
        print("🚀 Outlier detection complete.")
    
    return df

def clean_dataset(df):
    """Apply all cleaning functions to a dataset"""
    print("\nApplying comprehensive cleaning to a dataset...")
    
    df = standardize_column_names(df)
    df = clean_city_names(df)
    df = clean_and_convert_data_types(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = detect_and_handle_outliers(df)
    
    print("🚀 Comprehensive cleaning complete.")
    return df

def merge_datasets(df1, df2):
    """Merge the two datasets"""
    print("\n🚀 MERGING DATASETS...")
    
    # Ensure both datasets have the same structure
    df1_clean = clean_dataset(df1.copy())
    df2_clean = clean_dataset(df2.copy())
    
    # Add source identifier
    df1_clean['data_source'] = 'dataset_1'
    df2_clean['data_source'] = 'dataset_2'
    
    # Concatenate datasets
    merged_df = pd.concat([df1_clean, df2_clean], ignore_index=True)
    
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

def main():
    # Define file paths
    raw_data_path1 = "data/raw/lagos_air_pollution_health_data1.csv"
    raw_data_path2 = "data/raw/lagos_air_pollution_health_data2.csv"
    processed_data_path = "data/processed/cleaned_data.parquet"
    
    # Load and inspect data
    df1, df2 = load_and_inspect_data(raw_data_path1, raw_data_path2)
    
    if df1 is None or df2 is None:
        print("Failed to load data. Exiting.")
        return
    
    # Merge datasets
    merged_df = merge_datasets(df1, df2)
    
    # Save cleaned dataset
    try:
        merged_df.to_parquet(processed_data_path, index=False)
        print(f"🚀 Cleaned dataset saved as '{processed_data_path}'")
    except Exception as e:
        print(f"🚀 Error saving cleaned dataset: {e}")

if __name__ == "__main__":
    main()