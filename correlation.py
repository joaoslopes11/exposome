#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
SIMPLIFIED CORRELATION ANALYSIS FOR EXPOSOME RESULTS
=====================================================
This script works directly with your existing Excel file that already contains
the Exposome calculation results (the file you shared earlier).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_exposome_results(file_path):
    """Load the Exposome results Excel file"""
    print("="*80)
    print(" LOADING EXPOSOME RESULTS")
    print("="*80)
    
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    print(f"\n✓ File loaded: {file_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Check if the first row is a header row (sometimes Excel files have issues)
    if df.iloc[0, 0] == 'run_id,postal_code,country,age,sex,bmi,health_status,phenotype,region,smoking_status,secondhand_smoke,smoking_start_age,quit_years,occupation_exposures,occupation_years,protection_equipment,diet_pattern,fruit_veg_consumption,processed_meat,activity_level,family_history,respiratory_conditions':
        print("\n  ⚠️  Detected malformed CSV content in Excel. Re-reading as CSV...")
        # Extract the data properly
        data_str = df.iloc[:, 0].str.cat(sep='\n')
        from io import StringIO
        df = pd.read_csv(StringIO(data_str))
        print(f"  Recovered shape: {df.shape}")
    
    # Identify the actual data columns (skip metadata columns at the beginning)
    # Look for columns with risk scores
    risk_columns = ['Air_Pollution', 'Smoking', 'Occupational', 'Noise', 
                   'Temperature', 'Diet', 'Physical_Activity', 'Genetic', 
                   'Socioeconomic', 'CEI']
    
    # Find which risk columns exist
    available_risks = [col for col in risk_columns if col in df.columns]
    
    if not available_risks:
        # Try to find columns by position (sometimes column names are shifted)
        print("\n  Looking for risk columns by position...")
        # Check numeric columns that might be risk scores
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 10:
            # Assume the last 10 numeric columns are the risk scores
            available_risks = numeric_cols[-10:]
            print(f"  Found potential risk columns: {available_risks}")
    
    print(f"\n  Available risk components: {available_risks}")
    
    # Ensure CEI is included
    if 'CEI' not in available_risks and 'CEI' in df.columns:
        available_risks.append('CEI')
    
    return df, available_risks

def create_synthetic_wellbeing_index(df):
    """
    Create a realistic synthetic well-being index (0-1)
    Inversely correlated with risk scores
    """
    print("\n" + "="*80)
    print(" CREATING SYNTHETIC WELL-BEING INDEX")
    print("="*80)
    
    np.random.seed(42)  # For reproducibility
    
    # Find available risk columns
    risk_cols = [col for col in df.columns if col in 
                ['CEI', 'Air_Pollution', 'Smoking', 'Occupational', 
                 'Noise', 'Temperature', 'Diet', 'Physical_Activity', 
                 'Genetic', 'Socioeconomic']]
    
    if not risk_cols:
        raise ValueError("No risk columns found in the dataframe")
    
    # Create well-being inversely related to CEI if available
    if 'CEI' in risk_cols:
        # Normalize CEI to 0-1 range
        cei_normalized = (df['CEI'] - df['CEI'].min()) / (df['CEI'].max() - df['CEI'].min())
        wellbeing = 1 - cei_normalized
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.05, len(df))
        wellbeing = wellbeing + noise
        
    else:
        # Use weighted average of available risk components
        weights = {col: 1/len(risk_cols) for col in risk_cols}
        weighted_risk = sum(df[col] * weight for col, weight in weights.items())
        weighted_risk_normalized = (weighted_risk - weighted_risk.min()) / (weighted_risk.max() - weighted_risk.min())
        wellbeing = 1 - weighted_risk_normalized
        noise = np.random.normal(0, 0.05, len(df))
        wellbeing = wellbeing + noise
    
    # Clip to [0, 1] range
    wellbeing = np.clip(wellbeing, 0, 1)
    
    # Categorize well-being
    wellbeing_categories = pd.cut(wellbeing, 
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
    
    print(f"\n  Well-being statistics:")
    print(f"    Mean: {wellbeing.mean():.3f}")
    print(f"    Median: {wellbeing.median():.3f}")
    print(f"    Std: {wellbeing.std():.3f}")
    print(f"    Range: {wellbeing.min():.3f} - {wellbeing.max():.3f}")
    
    return wellbeing, wellbeing_categories

def calculate_correlations(df, wellbeing, risk_components):
    """Calculate Pearson and Spearman correlations"""
    print("\n" + "="*80)
    print(" CALCULATING CORRELATIONS")
    print("="*80)
    
    correlations = []
    
    for component in risk_components:
        if component not in df.columns:
            continue
            
        # Clean data - remove NaN values
        clean_data = df[[component]].copy()
        clean_data['wellbeing'] = wellbeing
        clean_data = clean_data.dropna()
        
        if len(clean_data) < 3:
            print(f"  Warning: Not enough data for {component}")
            continue
        
        x = clean_data[component].values
        y = clean_data['wellbeing'].values
        
        # Pearson correlation
        try:
            pearson_r, pearson_p = pearsonr(x, y)
        except:
            pearson_r, pearson_p = np.nan, np.nan
        
        # Spearman correlation
        try:
            spearman_r, spearman_p = spearmanr(x, y)
        except:
            spearman_r, spearman_p = np.nan, np.nan
        
        correlations.append({
            'Risk_Component': component,
            'Pearson_r': pearson_r,
            'Pearson_p_value': pearson_p,
            'Pearson_Significant': pearson_p < 0.05 if not np.isnan(pearson_p) else False,
            'Spearman_rho': spearman_r,
            'Spearman_p_value': spearman_p,
            'Spearman_Significant': spearman_p < 0.05 if not np.isnan(spearman_p) else False
        })
    
    df_corr = pd.DataFrame(correlations)
    df_corr['Pearson_r_abs'] = abs(df_corr['Pearson_r'])
    df_corr = df_corr.sort_values('Pearson_r_abs', ascending=False)
    
    print(f"\n  Calculated correlations for {len(df_corr)} components")
    
    return df_corr

def create_visualizations(df, wellbeing, correlations_df, output_folder='./correlation_analysis'):
    """Create comprehensive visualizations"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare data for visualizations
    df_vis = df.copy()
    df_vis['Wellbeing'] = wellbeing
    
    # Get risk components that exist
    risk_cols = [col for col in correlations_df['Risk_Component'].values if col in df_vis.columns]
    
    if not risk_cols:
        print("  No risk columns found for visualizations")
        return
    
    # 1. Correlation heatmap
    print("\n  Creating correlation heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Create correlation matrix
    corr_data = df_vis[risk_cols + ['Wellbeing']].copy()
    # Convert any non-numeric to numeric
    for col in corr_data.columns:
        corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')
    
    corr_matrix = corr_data.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5, 
                annot_kws={'size': 10})
    plt.title('Correlation Matrix: Exposome Risks vs Well-being', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_folder}/correlation_heatmap.png")
    
    # 2. Top correlations bar plot
    print("  Creating correlation bar plot...")
    plt.figure(figsize=(10, 6))
    
    top_corr = correlations_df.head(10)
    colors = ['red' if x < 0 else 'green' for x in top_corr['Pearson_r']]
    
    plt.barh(range(len(top_corr)), top_corr['Pearson_r'].values, color=colors)
    plt.yticks(range(len(top_corr)), top_corr['Risk_Component'].values)
    plt.xlabel('Pearson Correlation Coefficient', fontsize=12)
    plt.ylabel('Risk Components', fontsize=12)
    plt.title('Correlation between Risk Components and Well-being', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.3, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/correlation_barplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_folder}/correlation_barplot.png")
    
    # 3. Scatter plots for top 3 correlations
    print("  Creating scatter plots...")
    top_3 = correlations_df.head(3)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if len(top_3) < 3:
        fig, axes = plt.subplots(1, len(top_3), figsize=(15, 5))
    
    for idx, (_, row) in enumerate(top_3.iterrows()):
        if idx < len(axes):
            ax = axes[idx]
            component = row['Risk_Component']
            
            # Clean data
            plot_data = df_vis[[component, 'Wellbeing']].dropna()
            
            ax.scatter(plot_data[component], plot_data['Wellbeing'], alpha=0.6, s=50)
            
            # Add trend line
            if len(plot_data) > 1:
                z = np.polyfit(plot_data[component], plot_data['Wellbeing'], 1)
                p = np.poly1d(z)
                x_line = np.array([plot_data[component].min(), plot_data[component].max()])
                ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
                       label=f'r = {row["Pearson_r"]:.3f}')
            
            ax.set_xlabel(component, fontsize=11)
            ax.set_ylabel('Well-being Index', fontsize=11)
            ax.set_title(f'{component}\n(r = {row["Pearson_r"]:.3f}, p = {row["Pearson_p_value"]:.4f})', 
                        fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Top Correlations with Well-being', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_folder}/top_correlations_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_folder}/top_correlations_scatter.png")
    
    # 4. Boxplot of well-being by CEI categories
    print("  Creating boxplot...")
    if 'CEI' in df_vis.columns:
        plt.figure(figsize=(10, 6))
        
        # Create CEI categories
        df_vis['CEI_Category'] = pd.cut(df_vis['CEI'], 
                                        bins=[0, 20, 40, 60, 80, 100],
                                        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])
        
        # Boxplot
        df_vis.boxplot(column='Wellbeing', by='CEI_Category')
        plt.title('Well-being Distribution by CEI Risk Level', fontsize=14, fontweight='bold')
        plt.suptitle('')
        plt.xlabel('CEI Risk Level', fontsize=12)
        plt.ylabel('Well-being Index', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_folder}/wellbeing_by_cei.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: {output_folder}/wellbeing_by_cei.png")

def save_results(df, wellbeing, wellbeing_categories, correlations_df, output_folder='./correlation_analysis'):
    """Save all results to files"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['Wellbeing_Index'] = wellbeing
    results_df['Wellbeing_Level'] = wellbeing_categories
    
    # Save to Excel
    excel_file = f'{output_folder}/exposome_with_wellbeing.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Data_with_Wellbeing', index=False)
        correlations_df.to_excel(writer, sheet_name='Correlations', index=False)
        
        # Add summary statistics
        summary = pd.DataFrame({
            'Metric': [
                'Mean Well-being',
                'Median Well-being',
                'Std Well-being',
                'Min Well-being',
                'Max Well-being',
                'Number of Samples',
                'Best Correlation (strongest negative)',
                'Best Correlation Value',
                'Best Correlation p-value'
            ],
            'Value': [
                f"{wellbeing.mean():.3f}",
                f"{wellbeing.median():.3f}",
                f"{wellbeing.std():.3f}",
                f"{wellbeing.min():.3f}",
                f"{wellbeing.max():.3f}",
                len(df),
                correlations_df.iloc[0]['Risk_Component'] if len(correlations_df) > 0 else 'N/A',
                f"{correlations_df.iloc[0]['Pearson_r']:.3f}" if len(correlations_df) > 0 else 'N/A',
                f"{correlations_df.iloc[0]['Pearson_p_value']:.4f}" if len(correlations_df) > 0 else 'N/A'
            ]
        })
        summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
    
    print(f"\n  ✓ Results saved to: {excel_file}")
    
    # Also save as CSV
    csv_file = f'{output_folder}/exposome_with_wellbeing.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ CSV saved to: {csv_file}")
    
    return excel_file

def print_analysis_summary(df, wellbeing, correlations_df):
    """Print a comprehensive summary of the analysis"""
    print("\n" + "="*80)
    print(" CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total samples: {len(df)}")
    print(f"   Risk components analyzed: {len(correlations_df)}")
    
    print(f"\n📈 Well-being Statistics:")
    print(f"   Mean: {wellbeing.mean():.3f}")
    print(f"   Median: {wellbeing.median():.3f}")
    print(f"   Std: {wellbeing.std():.3f}")
    print(f"   Range: {wellbeing.min():.3f} - {wellbeing.max():.3f}")
    
    if len(correlations_df) > 0:
        print(f"\n📉 Top 3 Negative Correlations (Higher risk → Lower well-being):")
        negative_corr = correlations_df[correlations_df['Pearson_r'] < 0].head(3)
        for _, row in negative_corr.iterrows():
            significance = "✓" if row['Pearson_Significant'] else "✗"
            print(f"   {significance} {row['Risk_Component']}: r = {row['Pearson_r']:.3f} (p={row['Pearson_p_value']:.4f})")
        
        print(f"\n📈 Top 3 Positive Correlations (Higher risk → Higher well-being):")
        positive_corr = correlations_df[correlations_df['Pearson_r'] > 0].head(3)
        for _, row in positive_corr.iterrows():
            significance = "✓" if row['Pearson_Significant'] else "✗"
            print(f"   {significance} {row['Risk_Component']}: r = {row['Pearson_r']:.3f} (p={row['Pearson_p_value']:.4f})")
        
        print(f"\n🎯 Statistically Significant Correlations (p < 0.05):")
        significant = correlations_df[correlations_df['Pearson_Significant'] == True]
        if len(significant) > 0:
            for _, row in significant.iterrows():
                print(f"   • {row['Risk_Component']}: r = {row['Pearson_r']:.3f} (p={row['Pearson_p_value']:.4f})")
        else:
            print("   No statistically significant correlations found")
        
        print(f"\n💡 Key Insights:")
        strongest = correlations_df.iloc[0]
        if strongest['Pearson_r'] < -0.5:
            print(f"   • Very strong negative correlation with {strongest['Risk_Component']}")
            print(f"     → Higher {strongest['Risk_Component']} strongly associated with lower well-being")
        elif strongest['Pearson_r'] < -0.3:
            print(f"   • Moderate negative correlation with {strongest['Risk_Component']}")
        elif strongest['Pearson_r'] > 0.3:
            print(f"   • Positive correlation detected - this risk factor may have confounding factors")
        else:
            print("   • Weak correlations overall - well-being may be influenced by other factors")
    
    print("\n" + "="*80)

def main():
    """Main execution function"""
    print("="*80)
    print(" EXPOSOME CORRELATION ANALYSIS")
    print("="*80)
    
    # Configuration
    input_file = 'vigo_metadata_enhanced_exposome_complete.xlsx'
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"\n❌ File not found: {input_file}")
        print("\nPlease make sure the file exists in the current directory.")
        print("Looking for files with 'exposome' in the name:")
        for f in os.listdir('.'):
            if 'exposome' in f.lower() or 'sample' in f.lower():
                print(f"  - {f}")
        return None, None
    
    # Load data
    df, risk_components = load_exposome_results(input_file)
    
    if len(risk_components) == 0:
        print("\n❌ No risk components found in the file!")
        print("   Please check that your Excel file contains columns like:")
        print("   'CEI', 'Air_Pollution', 'Smoking', 'Diet', etc.")
        return None, None
    
    # Create synthetic well-being index
    wellbeing, wellbeing_categories = create_synthetic_wellbeing_index(df)
    
    # Calculate correlations
    correlations_df = calculate_correlations(df, wellbeing, risk_components)
    
    if len(correlations_df) == 0:
        print("\n❌ No correlations could be calculated!")
        return None, None
    
    # Create visualizations
    create_visualizations(df, wellbeing, correlations_df)
    
    # Save results
    save_results(df, wellbeing, wellbeing_categories, correlations_df)
    
    # Print summary
    print_analysis_summary(df, wellbeing, correlations_df)
    
    print("\n✅ Analysis completed successfully!")
    print("\n📁 Output folder: './correlation_analysis'")
    
    return df, correlations_df

# ============================================================================
# RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Create output directory
    os.makedirs('./correlation_analysis', exist_ok=True)
    
    # Run analysis
    df_results, corr_results = main()
    
    # Display sample results if successful
    if df_results is not None and corr_results is not None:
        print("\n📋 Sample of calculated results:")
        display_cols = ['run_id', 'CEI', 'Wellbeing_Index', 'Wellbeing_Level']
        available_cols = [col for col in display_cols if col in df_results.columns]
        if available_cols:
            print(df_results[available_cols].head(10))


# In[ ]:




