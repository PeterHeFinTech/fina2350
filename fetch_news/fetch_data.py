import requests
import json
import os
import time
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load API key from file
API_KEY_FILE = Path(__file__).resolve().parent / "api_key.txt"
if API_KEY_FILE.exists():
    API_KEY = API_KEY_FILE.read_text().strip()
else:
    API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not API_KEY:
        print("ERROR: api_key.txt not found and ALPHAVANTAGE_API_KEY env var not set")
        exit(1)
BASE_URL = "https://www.alphavantage.co/query"
OUT_FILE = Path(__file__).resolve().parent / "hardware_arch_shocks_alphavantage.json"
CSV_FILE = Path(__file__).resolve().parent / "nvda_sentiment_news.csv"
TARGET_TICKER = os.getenv("TARGET_TICKER", "NVDA").upper()
RELEVANCE_THRESHOLD = float(os.getenv("NVDA_RELEVANCE_THRESHOLD", "0.3"))

MAX_EVENTS = int(os.getenv("MAX_EVENTS", "20"))
SLEEP_BETWEEN_CALLS = float(os.getenv("SLEEP_SECONDS", "12"))

EVENTS = [
    {"event_date": "2023-06", "event_title": "AMD Unveils MI300X", "tickers": "AMD,NVDA", "topics": "technology"},
    {"event_date": "2023-12", "event_title": "Google Cloud TPU v5p", "tickers": "GOOG,GOOGL", "topics": "technology"},
    {"event_date": "2024-03", "event_title": "NVIDIA Blackwell (B200) Announcement", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2024-04", "event_title": "Intel Launches Gaudi 3", "tickers": "INTC,NVDA", "topics": "technology"},
    {"event_date": "2024-08", "event_title": "Groq Secures $640M (Series D)", "tickers": "NVDA,AMD,INTC", "topics": "technology"},
    {"event_date": "2024-08", "event_title": "Microsoft Maia 100 Detailed", "tickers": "MSFT,NVDA", "topics": "technology"},
    {"event_date": "2024-10", "event_title": "Meta MTIA Next-Gen", "tickers": "META,NVDA", "topics": "technology"},
    {"event_date": "2024-11", "event_title": "Hyperscaler H100 Saturation", "tickers": "NVDA,MSFT,META,AMZN", "topics": "technology"},
    {"event_date": "2025-01", "event_title": "NVIDIA Vera Rubin NVL72 Samples", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-03", "event_title": "Cerebras AI Inference Cloud", "tickers": "NVDA,AMD,INTC", "topics": "technology"},
    {"event_date": "2025-03", "event_title": "NVIDIA Blackwell Ultra Announced", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-04", "event_title": "Blackwell Reaches Consumers (RTX 50)", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-05", "event_title": "NVIDIA Enterprise AI Factories", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-09", "event_title": "Groq Hits $6.9B Valuation", "tickers": "NVDA,AMD", "topics": "technology"},
    {"event_date": "2025-10", "event_title": "OpenAI Custom Chip Finalized", "tickers": "NVDA,AMD,TSM", "topics": "technology"},
    {"event_date": "2025-11", "event_title": "Cerebras Withdraws IPO", "tickers": "NVDA,AMD,INTC", "topics": "technology"},
    {"event_date": "2025-12", "event_title": "NVIDIA Acquires Groq", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2026-02", "event_title": "Meta Expands Vera Rubin Clusters", "tickers": "META,NVDA", "topics": "technology"},
    {"event_date": "2026-03", "event_title": "Akamai Blackwell Edge Deployment", "tickers": "AKAM,NVDA", "topics": "technology"},
    {"event_date": "2026-03", "event_title": "NVIDIA GTC 2026 Vera Rubin Launch", "tickers": "NVDA", "topics": "technology"},
]


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_publish_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def extract_ticker_sentiment(item: dict, ticker: str) -> tuple[float | None, float | None]:
    for ticker_data in item.get("ticker_sentiment", []):
        if str(ticker_data.get("ticker", "")).upper() == ticker.upper():
            ticker_score = to_float(ticker_data.get("ticker_sentiment_score"))
            relevance = to_float(ticker_data.get("relevance_score"))
            return ticker_score, relevance
    return None, None


def month_window(event_date: str) -> tuple[str, str]:
    # event_date accepts YYYY-MM or YYYY-MM-DD; we search +/- 45 days around anchor date
    if len(event_date) == 7:
        anchor = datetime.strptime(event_date + "-15", "%Y-%m-%d")
    else:
        anchor = datetime.strptime(event_date, "%Y-%m-%d")

    start = anchor - timedelta(days=45)
    end = anchor + timedelta(days=45)
    return start.strftime("%Y%m%dT0000"), end.strftime("%Y%m%dT2359")


def fetch_event_news(event: dict) -> dict:
    time_from, time_to = month_window(event["event_date"])

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": event["tickers"],
        "topics": event["topics"],
        "time_from": time_from,
        "time_to": time_to,
        "sort": "RELEVANCE",
        "limit": "50",
        "apikey": API_KEY,
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    payload = response.json()

    # Alpha Vantage rate limit message
    if "Note" in payload:
        return {
            "event_date": event["event_date"],
            "event_title": event["event_title"],
            "time_from": time_from,
            "time_to": time_to,
            "tickers": event["tickers"],
            "error": payload.get("Note"),
            "articles": [],
        }

    feed = payload.get("feed", []) if isinstance(payload, dict) else []
    articles = []
    for item in feed:
        nvda_sentiment_score, nvda_relevance = extract_ticker_sentiment(item, TARGET_TICKER)
        if nvda_relevance is None or nvda_relevance <= RELEVANCE_THRESHOLD:
            continue

        articles.append(
            {
                "publish_date": format_publish_date(item.get("time_published")),
                "title": item.get("title"),
                "source": item.get("source"),
                "nvda_relevance": nvda_relevance,
                "nvda_sentiment_score": nvda_sentiment_score,
                "overall_sentiment_score": to_float(item.get("overall_sentiment_score")),
                "url": item.get("url"),
                "summary": item.get("summary"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
            }
        )

    return {
        "event_date": event["event_date"],
        "event_title": event["event_title"],
        "time_from": time_from,
        "time_to": time_to,
        "tickers": event["tickers"],
        "article_count": len(articles),
        "articles": articles,
    }


def main() -> None:
    selected_events = EVENTS[:MAX_EVENTS]

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Alpha Vantage NEWS_SENTIMENT",
        "events": [],
    }

    for idx, event in enumerate(selected_events, start=1):
        print(f"[{idx}/{len(selected_events)}] {event['event_title']}")
        event_result = fetch_event_news(event)
        result["events"].append(event_result)
        time.sleep(SLEEP_BETWEEN_CALLS)

    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUT_FILE}")

    csv_fields = [
        "publish_date",
        "title",
        "source",
        "nvda_relevance",
        "overall_sentiment_score",
        "url",
        "summary",
    ]

    csv_rows = []
    for event in result["events"]:
        for article in event.get("articles", []):
            csv_rows.append({field: article.get(field) for field in csv_fields})

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(
        f"Saved: {CSV_FILE} (rows={len(csv_rows)}, ticker={TARGET_TICKER}, relevance>{RELEVANCE_THRESHOLD})"
    )


if __name__ == "__main__":
    main()