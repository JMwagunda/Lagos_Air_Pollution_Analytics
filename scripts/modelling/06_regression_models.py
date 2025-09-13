"""
Regression modeling module for Lagos Air Pollution Analysis Project.
Builds and evaluates regression models to predict respiratory cases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def prepare_optimized_modeling_data(df):
    """Prepare data for machine learning models using core features only"""
    print("\n🚀 PREPARING OPTIMIZED MODELING DATA...")
    
    # Define core features based on importance analysis
    core_features = [
        'high_respiratory_risk',
        'cases_per_thousand',
        'population_density',
        'pm2_5',  # Primary pollution metric
        'pm10',   # Secondary pollution metric
        'weather_humidity',  # Affects pollution dispersion
    ]
    
    # Verify features exist in dataset
    available_features = [col for col in core_features if col in df.columns]
    print(f"Using core features: {available_features}")
    
    # Target variable
    target_col = 'respiratory_cases'
    if target_col not in df.columns:
        print(f"🚀 Target variable '{target_col}' not found in dataset")
        return None, None, None, None
    
    # Create feature matrix and target vector
    X = df[available_features].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target values
    mask = ~y.isnull() & X.notnull().all(axis=1)
    X = X[mask]
    y = y[mask]
    
    print(f"🚀 Optimized modeling data prepared:")
    print(f" Features shape: {X.shape}")
    print(f" Target shape: {y.shape}")
    print(f" Features used: {available_features}")
    
    return X, y, available_features, df

def build_optimized_models(X, y, feature_names):
    """Build and evaluate optimized machine learning models"""
    print("\n BUILDING OPTIMIZED MODELS...")
    print("=" * 40)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize optimized models (faster versions)
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=10,     # Limit depth
            min_samples_split=5,
            random_state=42,
            n_jobs=-1         # Use all cores
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50,  # Reduced from 100
            max_depth=6,      # Limit depth
            learning_rate=0.1,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score (faster with 3-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        results[name] = {
            'model': model,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'predictions': y_pred
        }
        
        print(f" 🚀 {name} Results:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R²: {r2:.3f}")
        print(f"  CV RMSE: {cv_rmse:.3f}")
    
    # Feature importance analysis
    print("\n🚀 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 35)
    
    for name, result in results.items():
        if hasattr(result['model'], 'feature_importances_'):
            importances = result['model'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n{name} - Feature Importance:")
            print(feature_importance.to_string(index=False))
    
    return results, X_test, y_test

def create_model_visualizations(results, X_test, y_test):
    """Create visualizations comparing model performance"""
    print("\n🚀 CREATING MODEL VISUALIZATIONS...")
    
    # Model performance comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Extract metrics
    model_names = list(results.keys())
    rmse_scores = [results[name]['rmse'] for name in model_names]
    mae_scores = [results[name]['mae'] for name in model_names]
    r2_scores = [results[name]['r2'] for name in model_names]
    
    # RMSE comparison
    axes[0].bar(model_names, rmse_scores, color='lightblue', edgecolor='navy')
    axes[0].set_title('Root Mean Square Error (RMSE)')
    axes[0].set_ylabel('RMSE')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # R² comparison
    axes[1].bar(model_names, r2_scores, color='lightgreen', edgecolor='darkgreen')
    axes[1].set_title('R² Score')
    axes[1].set_ylabel('R²')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[2].bar(model_names, mae_scores, color='lightcoral', edgecolor='darkred')
    axes[2].set_title('Mean Absolute Error (MAE)')
    axes[2].set_ylabel('MAE')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Actual vs Predicted for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_predictions = results[best_model_name]['predictions']
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_predictions, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('Actual Respiratory Cases')
    plt.ylabel('Predicted Respiratory Cases')
    plt.title(f'Actual vs Predicted - {best_model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² and RMSE to the plot
    r2_best = results[best_model_name]['r2']
    rmse_best = results[best_model_name]['rmse']
    plt.text(0.05, 0.95, f'R² = {r2_best:.3f}\nRMSE = {rmse_best:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name

def save_best_model(results, best_model_name):
    """Save the best performing model"""
    import joblib
    
    best_model = results[best_model_name]['model']
    model_path = f"outputs/models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
    
    try:
        joblib.dump(best_model, model_path)
        print(f"🚀 Best model ({best_model_name}) saved to {model_path}")
    except Exception as e:
        print(f"🚀 Error saving model: {e}")

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
    
    # Prepare modeling data
    X, y, feature_names, model_df = prepare_optimized_modeling_data(df)
    
    if X is None:
        print("Modeling data preparation failed. Cannot proceed with modeling.")
        return
    
    # Build and evaluate models
    results, X_test, y_test = build_optimized_models(X, y, feature_names)
    
    # Create model visualizations
    best_model_name = create_model_visualizations(results, X_test, y_test)
    
    # Save the best model
    save_best_model(results, best_model_name)
    
    print(f"\n🚀 Best performing model: {best_model_name}")
    print("🚀 Regression modeling complete.")

if __name__ == "__main__":
    main()