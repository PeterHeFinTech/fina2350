import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# --- SETUP ---
BASE_DIR = Path(__file__).resolve().parent
# Make sure this matches the filename from your previous step
DATA_FILE = BASE_DIR / "nvda_sentiment_regression_results.csv"
OUTPUT_IMAGE = BASE_DIR / "car1_regression_plot.png"

def generate_regression_plot():
    # 1. Load the data
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found. Please run the CAR calculation script first.")
        return
    
    df = pd.read_csv(DATA_FILE)
    
    # Ensure we have the right columns
    # We use Sentiment_Score as X and CAR_1 as Y
    x = df['sentiment_score']
    y = df['CAR_1']
    
    # 2. Calculate Stats for the legend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # 3. Create the Plot
    plt.figure(figsize=(10, 7), facecolor='white')
    sns.set_style("whitegrid")
    
    # Draw the regression line and scatter points
    plot = sns.regplot(
        x='sentiment_score', 
        y='CAR_1', 
        data=df, 
        scatter_kws={'s': 80, 'alpha': 0.6, 'color': '#1f77b4'},
        line_kws={'color': '#e74c3c', 'lw': 2},
        label=f"Regression Line (p={p_value:.4f})"
    )
    


    # 5. Add Labels and Stats Box
    plt.title('Impact of AI Model Sentiment on NVDA Abnormal Returns (CAR_1)', fontsize=14, pad=15)
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Cumulative Abnormal Return (Day 0 to 1)', fontsize=12)
    
    # Add a horizontal line at Y=0 to show the baseline market performance
    plt.axhline(0, color='black', lw=1, ls='--')

    stats_text = (f"R-Squared: {r_value**2:.4f}\n"
                  f"P-Value: {p_value:.4f}\n"
                  f"Beta: {slope:.4f}")
    
    plt.gca().text(0.05, 0.05, stats_text, transform=plt.gca().transAxes,
                   fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.legend()
    plt.tight_layout()
    
    # Save and Show
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved successfully to: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    generate_regression_plot()