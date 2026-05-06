import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "nvda_sentiment_regression_results.csv"
OUTPUT_IMAGE = BASE_DIR / "all_car_windows_regression_uniform.png"

def generate_multi_regression_plots():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found.")
        return
    
    df = pd.read_csv(DATA_FILE)
    
    # Identify the correct column names
    sent_col = 'sentiment_score' if 'sentiment_score' in df.columns else 'Sentiment_Score'
    car_windows = ['CAR_minus_1', 'CAR_0', 'CAR_1', 'CAR_2']
    
    # --- STEP 1: CALCULATE GLOBAL Y-LIMITS ---
    # We find the min and max across all columns to keep the scale identical
    y_min = df[car_windows].min().min()
    y_max = df[car_windows].max().max()
    
    # Add a 10% buffer so points aren't touching the edge of the frame
    padding = (y_max - y_min) * 0.1
    y_limit_bottom = y_min - padding
    y_limit_top = y_max + padding

    # --- STEP 2: PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    axes = axes.flatten()
    sns.set_style("whitegrid")

    for i, car_col in enumerate(car_windows):
        if car_col not in df.columns:
            continue
            
        correlation = df[sent_col].corr(df[car_col])
        
        sns.regplot(
            x=sent_col, 
            y=car_col, 
            data=df, 
            ax=axes[i],
            scatter_kws={'s': 50, 'alpha': 0.5, 'color': '#1f77b4'},
            line_kws={'color': '#e74c3c', 'lw': 2}
        )
        
        # Apply the uniform Y-limits
        axes[i].set_ylim(y_limit_bottom, y_limit_top)
        
        axes[i].axhline(0, color='black', lw=0.8, ls='--')
        axes[i].set_title(f'Window: {car_col}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Sentiment Score')
        axes[i].set_ylabel('Abnormal Return')
        
        axes[i].text(0.05, 0.92, f'Corr (r): {correlation:.4f}', 
                     transform=axes[i].transAxes, fontsize=11, 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('NVIDIA Sentiment vs. Abnormal Returns (Uniform Scaling)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Uniform plot saved to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    generate_multi_regression_plots()