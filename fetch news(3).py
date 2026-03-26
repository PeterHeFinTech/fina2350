import requests
import pandas as pd
import json

# ==========================================
# 1. Core Parameter Setup (Example: Category C Policy Shock)
# ==========================================
API_KEY = '##'  # Replace with your team's Alpha Vantage API Key
TICKER = 'NVDA'

# Event day T=0 is 2023-10-17. Setting the event window to [-1, +2]
TIME_FROM = '20230321T0000'
TIME_TO = '20230324T2359'

# Construct the API request URL (limit=1000 ensures we capture all news within the window)
url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={TICKER}&time_from={TIME_FROM}&time_to={TIME_TO}&limit=1000&apikey={API_KEY}'

print(f"Fetching news data for {TICKER} from {TIME_FROM} to {TIME_TO}...")

# ==========================================
# 2. Send Request and Fetch Data
# ==========================================
response = requests.get(url)
data = response.json()

# Check for API rate limits or errors
if 'feed' not in data:
    print("API request failed or rate limit reached. Please check your API Key or try again later.")
    print("API Response:", data)
    exit()

articles = data['feed']
print(f"Fetched {len(articles)} raw articles mentioning {TICKER}.")

# ==========================================
# 3. Data Cleaning: Extracting NVDA-specific Sentiment Scores
# ==========================================
clean_data = []

for article in articles:
    # Initialize variables
    nvda_score = None
    nvda_relevance = None
    
    # Search for NVDA in the ticker list of each article
    for ticker_data in article.get('ticker_sentiment', []):
        if ticker_data['ticker'] == TICKER:
            nvda_score = float(ticker_data['ticker_sentiment_score'])
            nvda_relevance = float(ticker_data['relevance_score'])
            break  # Exit inner loop once NVDA is found
    
    # Core filtering logic: Must contain NVDA with a relevance score > 0.3 (filtering out noise)
    if nvda_score is not None and nvda_relevance > 0.3:
        clean_data.append({
            'Published_Time': article.get('time_published'),
            'Title': article.get('title'),
            'Source': article.get('source'),
            'NVDA_Relevance': nvda_relevance,
            'NVDA_Sentiment_Score': nvda_score,
            'URL': article.get('url')
        })

# ==========================================
# 4. Convert to DataFrame and Export
# ==========================================
df = pd.DataFrame(clean_data)

if not df.empty:
    # Format the timestamp for better readability
    df['Published_Time'] = pd.to_datetime(df['Published_Time'], format='%Y%m%dT%H%M%S')
    
    print(f"\nCleaning complete! Kept {len(df)} high-quality and highly relevant articles for {TICKER}.")
    print(df[['Published_Time', 'Title', 'NVDA_Sentiment_Score']].head())
    
    # Export to CSV file, uniquely named with the event date for standardization
    csv_filename = 'NVDA_Event_CatC_03_FLIOpenLetter_20230322.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ Data successfully saved to: {csv_filename}")
else:
    print("\n⚠️ No articles met the relevance threshold. Try lowering the relevance score limit.")