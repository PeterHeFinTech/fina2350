import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load the Dataset
# ==========================================
csv_filename = "nvda_sentiment_cat_c_validated.csv"
try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"Error: '{csv_filename}' not found. Please ensure the dual-engine fetching script ran successfully.")
    exit()

# Clean the data by dropping rows with missing values in the target columns to ensure rigorous calculation
df_clean = df.dropna(subset=['nvda_sentiment_score', 'finbert_local_score'])

# ==========================================
# 2. Core Logic: Calculate Pearson Correlation
# ==========================================
# The .corr() function calculates the Pearson correlation coefficient (r) by default
correlation = df_clean['nvda_sentiment_score'].corr(df_clean['finbert_local_score'])

print("📊 Statistical Analysis Complete!")
print(f"Pearson Correlation Coefficient (r) between AlphaVantage API and Local FinBERT: {correlation:.4f}")

if correlation > 0.7:
    print("💡 Conclusion: Highly positive correlation! This mathematically validates the robustness and reliability of the commercial API's sentiment scoring.")

# ==========================================
# 3. Data Visualization: Scatter Plot with Trend Line
# ==========================================
# Set high resolution and figure size suitable for academic reports or presentations
plt.figure(figsize=(9, 7), dpi=300)

# Use seaborn to create a scatter plot and automatically fit a linear regression trend line
sns.regplot(
    x='finbert_local_score', 
    y='nvda_sentiment_score', 
    data=df_clean, 
    scatter_kws={'alpha': 0.7, 'color': '#2b5b84', 's': 60}, 
    line_kws={'color': '#d9534f', 'linewidth': 2, 'label': f'Trend Line (r = {correlation:.2f})'}
)

# Add title and axis labels
plt.title("Robustness Check: API Sentiment vs. Local FinBERT Model", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Local FinBERT Score (Baseline NLP)", fontsize=12, fontweight='bold')
plt.ylabel("Alpha Vantage API Score (ABSA Engine)", fontsize=12, fontweight='bold')

# Add grid lines for better readability
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11, loc='upper left')

# Add baseline reference lines (crosshairs) to distinguish bullish vs. bearish territories
plt.axhline(0, color='black', linewidth=1.2, linestyle='-')
plt.axvline(0, color='black', linewidth=1.2, linestyle='-')

# Adjust layout to prevent text clipping
plt.tight_layout()

# Export the plot as a high-resolution image
output_img = "correlation_robustness_check.png"
plt.savefig(output_img)
print(f"\n✅ Scatter plot generated successfully! Saved as high-resolution image: {output_img}")