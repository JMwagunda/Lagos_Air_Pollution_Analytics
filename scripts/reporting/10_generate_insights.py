"""
Insights generation module for Lagos Air Pollution Analysis Project.
Tests hypotheses and generates key insights from the analysis.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def test_hypotheses(df):
    """Test the research hypotheses"""
    print("\n🚀 HYPOTHESIS TESTING")
    print("=" * 30)
    
    hypothesis_results = {}
    
    # H1: Higher PM2.5 levels correlate with more respiratory cases
    if 'pm2_5' in df.columns and 'respiratory_cases' in df.columns:
        corr_coef, p_value = pearsonr(df['pm2_5'].dropna(), df['respiratory_cases'].dropna())
        
        hypothesis_results['H1'] = {
            'hypothesis': 'Higher PM2.5 levels correlate with more respiratory cases',
            'correlation': corr_coef,
            'p_value': p_value,
            'result': 'SUPPORTED' if corr_coef > 0 and p_value < 0.05 else 'NOT SUPPORTED',
            'interpretation': f"Correlation coefficient: {corr_coef:.3f}, p-value: {p_value:.3f}"
        }
        
        print(f"H1: {hypothesis_results['H1']['result']}")
        print(f" {hypothesis_results['H1']['interpretation']}")
    
    # H2: Cities with higher industrial indices have worse air quality
    if 'industrial_activity_index' in df.columns and 'pollution_index' in df.columns:
        corr_coef, p_value = pearsonr(df['industrial_activity_index'].dropna(), 
                                      df['pollution_index'].dropna())
        
        hypothesis_results['H2'] = {
            'hypothesis': 'Cities with higher industrial indices have worse air quality',
            'correlation': corr_coef,
            'p_value': p_value,
            'result': 'SUPPORTED' if corr_coef > 0 and p_value < 0.05 else 'NOT SUPPORTED',
            'interpretation': f"Correlation coefficient: {corr_coef:.3f}, p-value: {p_value:.3f}"
        }
        
        print(f"H2: {hypothesis_results['H2']['result']}")
        print(f" {hypothesis_results['H2']['interpretation']}")
    
    # H3: Harmattan season shows spikes in PM10 and respiratory cases
    if 'is_harmattan' in df.columns and 'pm10' in df.columns and 'respiratory_cases' in df.columns:
        harmattan_pm10 = df[df['is_harmattan'] == 1]['pm10'].mean()
        non_harmattan_pm10 = df[df['is_harmattan'] == 0]['pm10'].mean()
        harmattan_cases = df[df['is_harmattan'] == 1]['respiratory_cases'].mean()
        non_harmattan_cases = df[df['is_harmattan'] == 0]['respiratory_cases'].mean()
        
        # Perform t-tests
        t_stat_pm10, p_val_pm10 = ttest_ind(
            df[df['is_harmattan'] == 1]['pm10'].dropna(),
            df[df['is_harmattan'] == 0]['pm10'].dropna()
        )
        
        t_stat_cases, p_val_cases = ttest_ind(
            df[df['is_harmattan'] == 1]['respiratory_cases'].dropna(),
            df[df['is_harmattan'] == 0]['respiratory_cases'].dropna()
        )
        
        pm10_supported = 'SUPPORTED' if p_val_pm10 < 0.05 and harmattan_pm10 > non_harmattan_pm10 else 'NOT SUPPORTED'
        cases_supported = 'SUPPORTED' if p_val_cases < 0.05 and harmattan_cases > non_harmattan_cases else 'NOT SUPPORTED'
        
        hypothesis_results['H3'] = {
            'hypothesis': 'Harmattan season shows spikes in PM10 and respiratory cases',
            'pm10_result': pm10_supported,
            'cases_result': cases_supported,
            'pm10_p_value': p_val_pm10,
            'cases_p_value': p_val_cases,
            'interpretation': f"PM10: {pm10_supported} (p={p_val_pm10:.3f}), Cases: {cases_supported} (p={p_val_cases:.3f})"
        }
        
        print(f"H3: {hypothesis_results['H3']['interpretation']}")
    
    # H4: Population density correlates with respiratory cases
    if 'population_density' in df.columns and 'respiratory_cases' in df.columns:
        corr_coef, p_value = pearsonr(df['population_density'].dropna(), 
                                      df['respiratory_cases'].dropna())
        
        hypothesis_results['H4'] = {
            'hypothesis': 'Population density correlates with respiratory cases',
            'correlation': corr_coef,
            'p_value': p_value,
            'result': 'SUPPORTED' if abs(corr_coef) > 0 and p_value < 0.05 else 'NOT SUPPORTED',
            'interpretation': f"Correlation coefficient: {corr_coef:.3f}, p-value: {p_value:.3f}"
        }
        
        print(f"H4: {hypothesis_results['H4']['result']}")
        print(f" {hypothesis_results['H4']['interpretation']}")
    
    return hypothesis_results

def generate_key_insights(df, hypothesis_results):
    """Generate key insights from the analysis"""
    print("\n🚀 GENERATING KEY INSIGHTS")
    print("=" * 30)
    
    insights = []
    
    # Insight 1: Pollution levels
    if 'pm2_5' in df.columns:
        avg_pm25 = df['pm2_5'].mean()
        max_pm25 = df['pm2_5'].max()
        min_pm25 = df['pm2_5'].min()
        
        insight = f"Average PM2.5 level in Lagos is {avg_pm25:.2f} µg/m³, ranging from {min_pm25:.2f} to {max_pm25:.2f} µg/m³."
        insights.append(insight)
        print(f"1. {insight}")
    
    # Insight 2: Health impact
    if 'respiratory_cases' in df.columns:
        avg_cases = df['respiratory_cases'].mean()
        max_cases = df['respiratory_cases'].max()
        
        insight = f"Average respiratory cases recorded is {avg_cases:.2f}, with a maximum of {max_cases:.2f} cases."
        insights.append(insight)
        print(f"2. {insight}")
    
    # Insight 3: Seasonal patterns
    if 'season' in df.columns and 'pm2_5' in df.columns:
        seasonal_pm25 = df.groupby('season')['pm2_5'].mean()
        max_season = seasonal_pm25.idxmax()
        min_season = seasonal_pm25.idxmin()
        
        insight = f"PM2.5 levels are highest in {max_season} season ({seasonal_pm25[max_season]:.2f} µg/m³) and lowest in {min_season} season ({seasonal_pm25[min_season]:.2f} µg/m³)."
        insights.append(insight)
        print(f"3. {insight}")
    
    # Insight 4: City comparison
    if 'city' in df.columns and 'pm2_5' in df.columns:
        city_pm25 = df.groupby('city')['pm2_5'].mean()
        max_city = city_pm25.idxmax()
        min_city = city_pm25.idxmin()
        
        insight = f"Among the cities studied, {max_city} has the highest PM2.5 levels ({city_pm25[max_city]:.2f} µg/m³), while {min_city} has the lowest ({city_pm25[min_city]:.2f} µg/m³)."
        insights.append(insight)
        print(f"4. {insight}")
    
    # Insight 5: Hypothesis summary
    if hypothesis_results:
        supported_count = sum(1 for h in hypothesis_results.values() 
                            if h.get('result', 'NOT SUPPORTED') == 'SUPPORTED')
        total_count = len(hypothesis_results)
        
        insight = f"Out of {total_count} hypotheses tested, {supported_count} were supported by the data."
        insights.append(insight)
        print(f"5. {insight}")
    
    return insights

def create_insights_report(hypothesis_results, insights):
    """Create a comprehensive insights report"""
    print("\n🚀 CREATING INSIGHTS REPORT")
    print("=" * 30)
    
    # Create a report dataframe
    report_data = []
    
    # Add hypothesis results
    for h_key, h_value in hypothesis_results.items():
        if h_key == 'H3':
            report_data.append({
                'Category': 'Hypothesis',
                'Item': h_value['hypothesis'],
                'Result': f"PM10: {h_value['pm10_result']}, Cases: {h_value['cases_result']}",
                'Details': h_value['interpretation']
            })
        else:
            report_data.append({
                'Category': 'Hypothesis',
                'Item': h_value['hypothesis'],
                'Result': h_value['result'],
                'Details': h_value['interpretation']
            })
    
    # Add insights
    for i, insight in enumerate(insights, 1):
        report_data.append({
            'Category': 'Key Insight',
            'Item': f'Insight {i}',
            'Result': '',
            'Details': insight
        })
    
    # Create dataframe
    report_df = pd.DataFrame(report_data)
    
    # Save report
    report_path = 'outputs/reports/insights_report.csv'
    try:
        report_df.to_csv(report_path, index=False)
        print(f"🚀 Insights report saved to {report_path}")
    except Exception as e:
        print(f"🚀 Error saving insights report: {e}")
    
    # Display report
    print("\n🚀 INSIGHTS REPORT")
    print("=" * 50)
    print(report_df.to_string(index=False))
    
    return report_df

def visualize_hypothesis_results(hypothesis_results):
    """Visualize hypothesis testing results"""
    print("\n🚀 VISUALIZING HYPOTHESIS RESULTS")
    
    # Prepare data for visualization
    h_results = []
    
    for h_key, h_value in hypothesis_results.items():
        if h_key == 'H3':
            # For H3, we have two sub-results
            h_results.append({
                'Hypothesis': f'{h_key} - PM10',
                'Result': 1 if h_value['pm10_result'] == 'SUPPORTED' else 0,
                'P-Value': h_value['pm10_p_value']
            })
            
            h_results.append({
                'Hypothesis': f'{h_key} - Cases',
                'Result': 1 if h_value['cases_result'] == 'SUPPORTED' else 0,
                'P-Value': h_value['cases_p_value']
            })
        else:
            h_results.append({
                'Hypothesis': h_key,
                'Result': 1 if h_value['result'] == 'SUPPORTED' else 0,
                'P-Value': h_value['p_value']
            })
    
    # Create dataframe
    h_df = pd.DataFrame(h_results)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Bar plot for results
    plt.subplot(1, 2, 1)
    colors = ['green' if r == 1 else 'red' for r in h_df['Result']]
    plt.bar(h_df['Hypothesis'], h_df['Result'], color=colors)
    plt.title('Hypothesis Test Results')
    plt.ylabel('Supported (1) / Not Supported (0)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Bar plot for p-values
    plt.subplot(1, 2, 2)
    plt.bar(h_df['Hypothesis'], h_df['P-Value'], color='skyblue')
    plt.axhline(y=0.05, color='red', linestyle='--', label='p=0.05 threshold')
    plt.title('Hypothesis P-Values')
    plt.ylabel('P-Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/figures/hypothesis_test_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("🚀 Hypothesis results visualization saved.")

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
    
    # Test hypotheses
    hypothesis_results = test_hypotheses(df)
    
    # Generate key insights
    insights = generate_key_insights(df, hypothesis_results)
    
    # Create insights report
    report_df = create_insights_report(hypothesis_results, insights)
    
    # Visualize hypothesis results
    visualize_hypothesis_results(hypothesis_results)
    
    print("\n🚀 Insights generation complete.")

if __name__ == "__main__":
    main()