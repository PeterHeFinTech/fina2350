import pandas as pd
from transformers import pipeline

# 1. Load the authoritative academic financial sentiment analysis model (FinBERT)
print("Downloading and loading the FinBERT model (this may take a few minutes on the first run)...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# 2. Load the data previously fetched from AlphaVantage
csv_filename = "nvda_sentiment_cat_c.csv"
df = pd.read_csv(csv_filename)

print(f"Successfully loaded {len(df)} news articles. Starting local FinBERT cross-validation...")

# 3. Define a function to convert FinBERT's text labels into specific numerical scores (-1.0 to 1.0)
def get_finbert_score(text):
    try:
        # FinBERT outputs a list like [{'label': 'positive', 'score': 0.85}]
        result = finbert(str(text))[0]
        label = result['label']
        confidence = result['score']
        
        if label == 'positive':
            return confidence        # Positive score for bullish sentiment
        elif label == 'negative':
            return -confidence       # Negative score for bearish sentiment
        else:
            return 0.0               # 0.0 for neutral sentiment
    except:
        return 0.0

# 4. Have FinBERT re-read all news titles and score them
# We save the calculated scores into a new column called "finbert_local_score"
df['finbert_local_score'] = df['title'].apply(get_finbert_score)

# 5. Display the comparison results and save the data
print("\n✅ Cross-validation complete! Comparing AlphaVantage scores with local FinBERT scores:")
print(df[['title', 'nvda_sentiment_score', 'finbert_local_score']].head())

# Export to a new CSV containing the cross-validated data
output_csv = "nvda_sentiment_cat_c_validated.csv"
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n📁 New dataset with dual validation saved to: {output_csv}")