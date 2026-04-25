from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
NEWS_FILE = BASE_DIR / "nvda_ai_sentiment_event_summary.csv"
NVDA_FILE = BASE_DIR / "NVDA_5Y.csv"
QQQ_FILE = BASE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "nvda_ai_sentiment_with_car.csv"


def load_price_series(path: Path, close_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[close_name] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["Date", close_name]].dropna().sort_values("Date")


def build_abnormal_returns() -> pd.DataFrame:
    nvda = load_price_series(NVDA_FILE, "Close_NVDA")
    qqq = load_price_series(QQQ_FILE, "Close_QQQ")
    merged = pd.merge(nvda, qqq, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    merged["r_nvda"] = merged["Close_NVDA"].pct_change()
    merged["r_qqq"] = merged["Close_QQQ"].pct_change()
    merged["ar"] = merged["r_nvda"] - merged["r_qqq"]
    return merged.dropna(subset=["ar"]).reset_index(drop=True)


def find_event_index(trading_dates: pd.Series, event_date: pd.Timestamp) -> int | None:
    pos = trading_dates.searchsorted(event_date)
    if pos >= len(trading_dates):
        return None
    return int(pos)


def compute_car(returns_df: pd.DataFrame, t_idx: int):
    start = t_idx - 1
    end = t_idx + 2
    if start < 0 or end >= len(returns_df):
        return None
    w = returns_df.loc[start:end, "ar"].reset_index(drop=True)
    return {
        "CAR-1": float(w.iloc[0]),
        "CAR0": float(w.iloc[0:2].sum()),
        "CAR1": float(w.iloc[0:3].sum()),
        "CAR2": float(w.iloc[0:4].sum()),
    }


def main():
    news = pd.read_csv(NEWS_FILE)
    if "event_date" not in news.columns:
        raise ValueError("nvda_ai_sentiment_event_summary.csv must contain publish_date")

    news["event_date"] = pd.to_datetime(news["event_date"], errors="coerce")
    returns_df = build_abnormal_returns()
    trading_dates = returns_df["Date"]

    rows = []
    for _, row in news.iterrows():
        if pd.isna(row.get("event_date")):
            continue
        event_date = row["event_date"].normalize()
        t_idx = find_event_index(trading_dates, event_date)
        if t_idx is None:
            continue
        car = compute_car(returns_df, t_idx)
        if car is None:
            continue
        out = row.to_dict()
        out.update(car)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved {OUT_FILE} rows={len(out_df)} cols={len(out_df.columns)}")


if __name__ == "__main__":
    main()
