import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_category_c_correlation(csv_path="category_C_car_sentiment.csv", out_img="cat_c_correlation_plots.png"):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please run the CAR calculation script first.")
        return

    # 2. Preprocessing
    cols = ["CAR-1", "CAR0", "CAR1", "CAR2", "sentiment"]
    df = df[cols].apply(pd.to_numeric, errors='coerce').dropna()

    # 3. Calculate Correlation Matrix
    corr_matrix = df.corr()
    print("\n" + "="*30)
    print("PEARSON CORRELATION MATRIX")
    print("="*30)
    print(corr_matrix[['sentiment']].sort_values(by='sentiment', ascending=False))
    print("="*30 + "\n")

    # 4. Plotting (4 Subplots with Regression Lines)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    y_cols = ["CAR-1", "CAR0", "CAR1", "CAR2"]

    for ax, y_col in zip(axes, y_cols):
        # Calculate specific correlation for the title
        r_value = df['sentiment'].corr(df[y_col])
        
        # Plot Scatter + Regression Line
        sns.regplot(
            data=df, x='sentiment', y=y_col, ax=ax, 
            scatter_kws={'alpha':0.6, 's':60, 'color':'#d62728'}, 
            line_kws={'color':'#1f77b4', 'lw':2},
            ci=None # Confidence Interval
        )
        
        ax.set_title(f"{y_col} vs Sentiment (r = {r_value:.3f})", fontsize=13, fontweight='bold')
        ax.set_xlabel("News Sentiment Score", fontsize=11)
        ax.set_ylabel(f"Abnormal Return ({y_col})", fontsize=11)
        ax.axhline(0, color='black', lw=1, ls='--')

    fig.suptitle("Category C: Correlation Analysis between Policy Sentiment and CAR", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save results
    fig.savefig(out_img, dpi=300)
    print(f"Visual analysis saved to: {out_img}")

if __name__ == "__main__":
    analyze_category_c_correlation()
    