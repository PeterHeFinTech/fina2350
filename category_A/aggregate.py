from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "nvda_sentiment_news_vader_cleaned.csv"
OUT_FILE = BASE_DIR / "nvda_ai_sentiment_event_summary.csv"

df = pd.read_csv(IN_FILE)
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

# Filter to AI-related only
df_ai = df[df["is_ai_related"] == True].copy()

# Aggregate by event_date_match (use publish_date if no event match)
df_ai["analysis_date"] = pd.to_datetime(df_ai["event_date_match"], errors="coerce").fillna(df_ai["publish_date"])
daily_sentiment = df_ai.groupby("analysis_date").agg(
    avg_ai_sentiment=("vader_compound", "mean"),
    num_ai_articles=("vader_compound", "count"),
    avg_nvda_relevance=("nvda_relevance", "mean"),
).reset_index().rename(columns={"analysis_date": "event_date"})

daily_sentiment.to_csv(OUT_FILE, index=False)
print(f"Saved AI sentiment summary: {OUT_FILE} (rows={len(daily_sentiment)})")