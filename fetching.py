import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('vader_lexicon')
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# Ensure VADER is downloaded
nltk.download('vader_lexicon', quiet=True)

def analyze_policy_impact(raw_input_csv, final_output_csv):
    # 1. Load the original raw data (category_C.csv)
    df = pd.read_csv(raw_input_csv)
    sia = SentimentIntensityAnalyzer()
    
    # 2. Define Policy & Regulatory Keywords (Category C Specific)
    policy_keywords = {
        # --- Restrictions & Bans ---
        'ban', 'restriction', 'restrict', 'curb', 'halt', 'stop', 'limit', 
        'export', 'control', 'sanction', 'blacklist', 'entity', 'prohibit',

        # --- Subsidies & Support ---
        'subsidy', 'subsidies', 'grant', 'funding', 'incentive', 'chips', 'act', 
        'support', 'investment', 'legislation', 'bill',

        # --- Government & Regulatory ---
        'biden', 'government', 'whitehouse', 'commerce', 'department', 'regulation', 
        'regulatory', 'antitrust', 'probe', 'investigation', 'china', 'beijing', 
        'official', 'policy', 'admin', 'administration'
    }

    def process_row(row):
        # GET RAW TEXT
        # Using Title + Summary for better keyword matching density
        raw_text = str(row['title']) + " " + str(row['summary'])
        
        # CREATE CLEAN TEXT for Relevance Matching
        clean_text = re.sub(r'[^A-Za-z0-9\s]+', '', raw_text).lower()
        
        # --- Relevance Logic (Category A Framework) ---
        words = clean_text.split()
        if not words: return 0.0, 0.0
        
        # Calculate how many policy-related words appear in the text
        matches = sum(1 for word in words if any(k == word for k in policy_keywords))
        relevance = round((matches / len(words)) * 100, 2)

        # --- Sentiment Logic (VADER) ---
        # Same as Category A, using raw text for punctuation/case sensitivity
        sentiment_score = sia.polarity_scores(raw_text)['compound']
        
        return relevance, sentiment_score

    # 3. Apply Trading Day Alignment (The 16:00 ET Cutoff)
    # This is essential for Category C to link correctly to Stock Returns (AR/CAR)
    def adjust_trading_date(date_str):
        # MarketAux returns UTC. UTC 20:00 = 16:00 ET (Market Close)
        dt = pd.to_datetime(date_str)
        if dt.hour >= 20:
            return (dt + timedelta(days=1)).strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d')

    print("Analyzing Policy Relevance and Sentiment...")
    
    # Apply processing logic
    results = df.apply(process_row, axis=1)
    df['policy_relevance_score'], df['sentiment_score'] = zip(*results)
    
    # Labeling
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'Positive' if x >= 0.05 else ('Negative' if x <= -0.05 else 'Neutral')
    )
    
    # Adjust publish date for Market Close
    df['publish_date'] = df['publish_date'].apply(adjust_trading_date)

    # 4. Final Aggregation (Aligned with Category A)
    # We group by the adjusted trading date to match daily Stock Returns
    final_report = df.groupby('publish_date').agg({
        'title': 'first',
        'policy_relevance_score': 'mean',
        'sentiment_score': 'mean',
        'source': 'first',
        'url': 'first'
    }).reset_index()

    # Sort by date for the Event Study
    final_report = final_report.sort_values(by='publish_date')

    # Rename columns to match the "Cleaned" format expects
    final_report.columns = ['publish_date', 'title', 'relevance_score', 'summary_sentiment_score', 'source', 'url']

    final_report.to_csv(final_output_csv, index=False)
    print(f"Category C report generated: {final_output_csv}")

# EXECUTION
analyze_policy_impact('category_C.csv', 'category_C_news_cleaned.csv')
