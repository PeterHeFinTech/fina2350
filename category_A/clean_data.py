from pathlib import Path
import re
import html
import pandas as pd

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import pos_tag


BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "nvda_sentiment_news.csv"
OUT_FILE = BASE_DIR / "nvda_sentiment_news_vader_cleaned.csv"


EVENTS = [
    {"event_date": "2022-11-30", "event_title": "ChatGPT Launch", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2023-03-14", "event_title": "GPT-4 Release", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2023-05-24", "event_title": "Nvidia Massive Guidance Beat", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2023-12-06", "event_title": "Google Gemini 1.0", "tickers": "GOOG,GOOGL", "topics": "technology"},
    {"event_date": "2024-02-15", "event_title": "OpenAI Sora Preview", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2024-03-18", "event_title": "Nvidia Blackwell Reveal", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2024-04-18", "event_title": "Meta Llama 3 Release", "tickers": "META", "topics": "technology"},
    {"event_date": "2024-12-09", "event_title": "OpenAI Sora Public Release", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2025-01-20", "event_title": "DeepSeek-R1 Shock", "tickers": "NVDA", "topics": "technology"},
    {"event_date": "2025-03-24", "event_title": "OpenAI Sora App Shutdown", "tickers": "MSFT", "topics": "technology"},
    {"event_date": "2025-09-30", "event_title": "Sora 2 / Social AI launch", "tickers": "MSFT", "topics": "technology"}
]


EVENT_KEYWORDS = {
    "ChatGPT Launch": ["chatgpt", "openai", "gpt"],
    "GPT-4 Release": ["gpt-4", "gpt 4", "openai"],
    "Nvidia Massive Guidance Beat": ["nvidia", "guidance", "ai demand", "data center", "gpu"],
    "Google Gemini 1.0": ["gemini", "google", "alphabet"],
    "OpenAI Sora Preview": ["sora", "openai", "text-to-video"],
    "Nvidia Blackwell Reveal": ["blackwell", "nvidia", "gpu", "ai chip", "accelerator"],
    "Meta Llama 3 Release": ["llama 3", "llama", "meta", "open-weight"],
    "OpenAI Sora Public Release": ["sora", "openai"],
    "DeepSeek-R1 Shock": ["deepseek", "deepseek-r1", "deepseek r1", "reasoning model"],
    "OpenAI Sora App Shutdown": ["sora", "openai"],
    "Sora 2 / Social AI launch": ["sora 2", "sora", "social ai", "openai"]
}


AI_KEYWORDS = sorted(set(
    [
        "artificial intelligence", "ai", "generative ai", "genai",
        "large language model", "llm", "foundation model", "transformer",
        "multimodal", "reasoning model", "text-to-video", "open-weight",
        "gpu", "ai chip", "accelerator", "data center", "ai demand"
    ] + [kw for kws in EVENT_KEYWORDS.values() for kw in kws]
))


def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("vader_lexicon", quiet=True)


def basic_clean(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\-\./]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("N"):
        return wordnet.NOUN
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    return " ".join(lemmas)


def remove_stopwords(text: str, stop_words: set) -> str:
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in stop_words]
    return " ".join(filtered)


def match_event(text: str) -> str | None:
    text_low = text.lower()
    for event_name, keywords in EVENT_KEYWORDS.items():
        if any(kw in text_low for kw in keywords):
            return event_name
    return None


def ai_keyword_hits(text: str) -> int:
    text_low = text.lower()
    return sum(1 for kw in AI_KEYWORDS if kw in text_low)


def vader_label(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def main():
    download_nltk_resources()

    df = pd.read_csv(IN_FILE)

    expected_cols = [
        "publish_date", "title", "source", "nvda_relevance", "nvda_sentiment_score",
        "overall_sentiment_score", "url", "summary"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["title"] = df["title"].fillna("").astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["source"] = df["source"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)

    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    df["raw_text"] = (df["title"].str.strip() + " " + df["summary"].str.strip()).str.strip()
    df = df[df["raw_text"].str.len() > 0].copy()

    df = df.drop_duplicates(subset=["url"], keep="first")
    df = df.drop_duplicates(subset=["publish_date", "title", "summary"], keep="first")
    df = df.reset_index(drop=True)

    df["text_clean"] = df["raw_text"].apply(basic_clean)
    df["text_lower"] = df["text_clean"].str.lower()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    df["text_no_stopwords"] = df["text_lower"].apply(lambda x: remove_stopwords(x, stop_words))
    df["text_lemmatized"] = df["text_no_stopwords"].apply(lambda x: lemmatize_text(x, lemmatizer))

    df["ai_keyword_hits"] = df["text_lower"].apply(ai_keyword_hits)
    df["is_ai_related"] = df["ai_keyword_hits"] > 0
    df["matched_event"] = df["text_lower"].apply(match_event)

    sia = SentimentIntensityAnalyzer()
    df["vader_neg"] = df["text_clean"].apply(lambda x: sia.polarity_scores(x)["neg"])
    df["vader_neu"] = df["text_clean"].apply(lambda x: sia.polarity_scores(x)["neu"])
    df["vader_pos"] = df["text_clean"].apply(lambda x: sia.polarity_scores(x)["pos"])
    df["vader_compound"] = df["text_clean"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["vader_label"] = df["vader_compound"].apply(vader_label)

    df["event_date_match"] = pd.NA
    event_date_map = {e["event_title"]: e["event_date"] for e in EVENTS}
    df.loc[df["matched_event"].notna(), "event_date_match"] = df.loc[
        df["matched_event"].notna(), "matched_event"
    ].map(event_date_map)

    output_cols = [
        "publish_date",
        "title",
        "source",
        "url",
        "summary",
        "nvda_relevance",
        "nvda_sentiment_score",
        "overall_sentiment_score",
        "raw_text",
        "text_clean",
        "text_lower",
        "text_no_stopwords",
        "text_lemmatized",
        "ai_keyword_hits",
        "is_ai_related",
        "matched_event",
        "event_date_match",
        "vader_neg",
        "vader_neu",
        "vader_pos",
        "vader_compound",
        "vader_label"
    ]

    out_df = df[output_cols].sort_values("publish_date")
    out_df.to_csv(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE}")
    print(f"Rows: {len(out_df)}")
    print(f"AI-related rows: {int(out_df['is_ai_related'].sum())}")
    print("Matched events:")
    print(out_df["matched_event"].value_counts(dropna=False))


if __name__ == "__main__":
    main()