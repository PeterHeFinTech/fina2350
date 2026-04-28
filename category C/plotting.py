import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_category_c_correlation(csv_path="category_C_car_sentiment.csv", out_img="cat_c_correlation_plots.png"):
    """
    Analyzes and visualizes the correlation between NVDA policy sentiment and Cumulative Abnormal Returns (CAR).
    Updated to match the latest CSV schema with fractional precision.
    """
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
        print(f"Status: Successfully loaded {csv_path}")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please ensure the CAR calculation script has been executed.")
        return

    # 2. Data Preprocessing & Cleaning
    # Mapping to exact column names in your latest CSV
    target_cols = ["CAR_minus_1", "CAR_0", "CAR_1", "CAR_2", "nvda_sentiment", "overall_sentiment"]
    
    # Ensure all analysis columns are numeric and drop missing values
    df_clean = df[target_cols].apply(pd.to_numeric, errors='coerce').dropna()

    # 3. Statistical Summary: Pearson Correlation
    corr_matrix = df_clean.corr()
    print("\n" + "="*50)
    print("PEARSON CORRELATION ANALYSIS (Target: nvda_sentiment)")
    print("="*50)
    # Focus on how nvda_sentiment relates to different CAR windows
    print(corr_matrix[['nvda_sentiment']].sort_values(by='nvda_sentiment', ascending=False))
    print("="*50 + "\n")

    # 4. Visualization: 4-Panel Regression Analysis
    # Using a professional academic style
    plt.style.use('ggplot') 
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()
    
    car_windows = ["CAR_minus_1", "CAR_0", "CAR_1", "CAR_2"]
    plot_color = '#d62728' # Professional red for scatter
    line_color = '#1f77b4' # Professional blue for regression

    for ax, y_col in zip(axes, car_windows):
        # Calculate specific Pearson R for the subplot title
        r_val = df_clean['nvda_sentiment'].corr(df_clean[y_col])
        
        # Plot Scatter + OLS Regression Line with 95% Confidence Interval
        sns.regplot(
            data=df_clean, 
            x='nvda_sentiment', 
            y=y_col, 
            ax=ax, 
            scatter_kws={'alpha':0.5, 's':50, 'color': plot_color}, 
            line_kws={'color': line_color, 'lw':2.5},
            ci=95 
        )
        
        # Formatting each subplot
        ax.set_title(f"{y_col} vs Policy Sentiment (r = {r_val:.3f})", fontsize=13, fontweight='bold')
        ax.set_xlabel("NVDA Sentiment Score (FinBERT)", fontsize=11)
        ax.set_ylabel(f"Abnormal Return ({y_col})", fontsize=11)
        
        # Add zero-lines for better quadrant visualization
        ax.axhline(0, color='black', lw=1, ls='--')
        ax.axvline(0, color='black', lw=1, ls='--')

    # Global Figure Title
    fig.suptitle("Category C: Regression Analysis of Regulatory Shocks on NVDA Stock Performance", 
                 fontsize=17, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save high-resolution output for report/presentation
    fig.savefig(out_img, dpi=300)
    print(f"Success: Analysis visualization saved as {out_img}")

if __name__ == "__main__":
    # Execute analysis
    analyze_category_c_correlation()