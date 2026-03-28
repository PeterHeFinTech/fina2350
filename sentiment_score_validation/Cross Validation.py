import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

print("==================================================")
print(" Executing FinBERT Cross-Validation (Title + Summary)")
print("==================================================")

# ==========================================
# 1. Load the Purified Dataset
# ==========================================
input_csv = "/Users/hetianqu/Documents/FINA2350/nvda_sentiment_news.csv"
try:
    df = pd.read_csv(input_csv)
    print(f" Loaded purified dataset: {len(df)} core events found.")
except FileNotFoundError:
    print(f" Error: '{input_csv}' not found. Please run the purification script first.")
    exit()

# ==========================================
# 2. Combine Title and Summary for Context
# ==========================================
print("🔗 Concatenating Title and Summary for deeper context...")
# Fill NaN with empty strings to avoid concatenation errors
df['combined_text'] = df['title'].astype(str) + " - " + df['summary'].fillna('').astype(str)

# ==========================================
# 3. Initialize Academic FinBERT Model
# ==========================================
print("⏳ Loading academic FinBERT model (ProsusAI/finbert)...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_finbert_score(text):
    try:
        # Truncate to 1500 chars to safely stay within BERT's 512 token limit
        safe_text = str(text)[:1500] 
        result = finbert(safe_text)[0]
        label = result['label']
        confidence = result['score']
        
        if label == 'positive': return round(confidence, 4)
        elif label == 'negative': return round(-confidence, 4)
        else: return 0.0
    except:
        return 0.0

# ==========================================
# 4. Apply FinBERT Scoring
# ==========================================
print(" FinBERT is analyzing the deep contextual text. Please wait...")
df['finbert_local_score'] = df['combined_text'].apply(get_finbert_score)

# ==========================================
# 5. Calculate Correlation & Plot Scatter Chart
# ==========================================
# Drop any potential NaN values in the score columns to ensure rigorous calculation
df_clean = df.dropna(subset=['nvda_sentiment_score', 'finbert_local_score'])

if len(df_clean) > 1:
    correlation = df_clean['nvda_sentiment_score'].corr(df_clean['finbert_local_score'])
    print(f"Statistical Analysis: Pearson correlation (r) = {correlation:.4f}")

    # Plotting the Robustness Check
    plt.figure(figsize=(9, 7), dpi=300)
    sns.regplot(
        x='finbert_local_score', 
        y='nvda_sentiment_score', 
        data=df_clean, 
        scatter_kws={'alpha': 0.8, 'color': '#2b5b84', 's': 100}, 
        line_kws={'color': '#d9534f', 'linewidth': 2, 'label': f'Trend Line (r = {correlation:.2f})'}
    )

    plt.title("Robustness Check: API vs. FinBERT (Purified Signals)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Local FinBERT Score (Title + Summary Context)", fontsize=12, fontweight='bold')
    plt.ylabel("Alpha Vantage API Score (ABSA Entity Isolation)", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11, loc='upper left')
    plt.axhline(0, color='black', linewidth=1.2, linestyle='-')
    plt.axvline(0, color='black', linewidth=1.2, linestyle='-')
    plt.tight_layout()

    output_img = "correlation_purified_signals.png"
    plt.savefig(output_img)
    print(f" Scatter plot saved successfully as: {output_img}")
else:
    print("Not enough data points to calculate correlation or plot.")

# ==========================================
# 6. Export the Final Regression Dataset
# ==========================================
# Drop the 'combined_text' helper column to keep the CSV extremely clean
df.drop(columns=['combined_text'], inplace=True, errors='ignore')

output_csv = "nvda_cat_c_final_validated.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')

print(f" Final Cross-Validated dataset saved to: {output_csv}")
print("==================================================")
print(" The Independent Variable (X) is now 100% ready for Ridge Regression!")
