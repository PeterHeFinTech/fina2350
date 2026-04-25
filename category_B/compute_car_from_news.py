from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PRICE_DIR = BASE_DIR.parent / "nvda_price_data"

NEWS_FILE = BASE_DIR / "category_B.csv"
NVDA_FILE = PRICE_DIR / "NVDA_5Y.csv"
QQQ_FILE = PRICE_DIR / "QQQ_5Y.csv"
OUT_FILE = BASE_DIR / "category_B_car_sentiment.csv"

# CAPM event-study settings
ESTIMATION_WINDOW = 120
ESTIMATION_GAP = 20
ANNUAL_RF_RATE = 0.02
TRADING_DAYS_PER_YEAR = 252


def load_price_series(path: Path, close_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df[close_name] = pd.to_numeric(df["Close"], errors="coerce")
    return df[["Date", close_name]].dropna().sort_values("Date")


def build_abnormal_returns() -> pd.DataFrame:
    nvda = load_price_series(NVDA_FILE, "Close_NVDA")
    qqq = load_price_series(QQQ_FILE, "Close_QQQ")

    merged = pd.merge(nvda, qqq, on="Date", how="inner").sort_values("Date")
    merged["r_nvda"] = merged["Close_NVDA"].pct_change()
    merged["r_qqq"] = merged["Close_QQQ"].pct_change()
    merged = merged.dropna(subset=["r_nvda", "r_qqq"]).reset_index(drop=True)
    return merged


def find_event_index(trading_dates: pd.Series, news_date: pd.Timestamp) -> int | None:
    pos = trading_dates.searchsorted(news_date)
    if pos >= len(trading_dates):
        return None
    return int(pos)


def estimate_capm_alpha_beta(
    returns_df: pd.DataFrame,
    t_idx: int,
    estimation_window: int = ESTIMATION_WINDOW,
    estimation_gap: int = ESTIMATION_GAP,
    daily_rf: float = 0.0,
) -> tuple[float | None, float | None]:
    est_end = t_idx - estimation_gap - 1
    est_start = est_end - estimation_window + 1

    if est_start < 0 or est_end < est_start:
        return None, None

    est = returns_df.loc[est_start:est_end, ["r_nvda", "r_qqq"]].dropna()
    if len(est) < 30:
        return None, None

    x = est["r_qqq"].to_numpy() - daily_rf
    y = est["r_nvda"].to_numpy() - daily_rf

    x_var = float(x.var())
    if x_var <= 1e-16:
        return None, None

    beta = float(np.cov(x, y, ddof=0)[0, 1] / x_var)
    alpha = float(y.mean() - beta * x.mean())
    return alpha, beta


def compute_car_row(
    returns_df: pd.DataFrame,
    t_idx: int,
    alpha: float,
    beta: float,
    daily_rf: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    start = t_idx - 1
    end = t_idx + 2

    if start < 0 or end >= len(returns_df):
        return None, None, None, None

    event_window = returns_df.loc[start:end, ["r_nvda", "r_qqq"]].reset_index(drop=True)
    expected_ret = daily_rf + alpha + beta * (event_window["r_qqq"] - daily_rf)
    ar_window = event_window["r_nvda"] - expected_ret

    car_m1 = float(ar_window.iloc[0])
    car_0 = float(ar_window.iloc[0:2].sum())
    car_1 = float(ar_window.iloc[0:3].sum())
    car_2 = float(ar_window.iloc[0:4].sum())
    return car_m1, car_0, car_1, car_2


def main() -> None:
    news = pd.read_csv(NEWS_FILE)
    news["publish_date"] = pd.to_datetime(news["publish_date"], errors="coerce")
    news["sentiment"] = pd.to_numeric(news["summary_sentiment_score"], errors="coerce")

    returns_df = build_abnormal_returns()
    trading_dates = returns_df["Date"]
    daily_rf = ANNUAL_RF_RATE / TRADING_DAYS_PER_YEAR

    rows = []
    for _, row in news.iterrows():
        if pd.isna(row["publish_date"]):
            continue

        event_date = row["publish_date"].normalize()
        t_idx = find_event_index(trading_dates, event_date)
        if t_idx is None:
            continue

        alpha, beta = estimate_capm_alpha_beta(
            returns_df,
            t_idx,
            estimation_window=ESTIMATION_WINDOW,
            estimation_gap=ESTIMATION_GAP,
            daily_rf=daily_rf,
        )
        if alpha is None or beta is None:
            continue

        car_m1, car_0, car_1, car_2 = compute_car_row(
            returns_df,
            t_idx,
            alpha=alpha,
            beta=beta,
            daily_rf=daily_rf,
        )
        if car_m1 is None:
            continue

        rows.append(
            {
                "CAR-1": car_m1,
                "CAR0": car_0,
                "CAR1": car_1,
                "CAR2": car_2,
                "sentiment": row["sentiment"],
            }
        )

    out_df = pd.DataFrame(rows, columns=["CAR-1", "CAR0", "CAR1", "CAR2", "sentiment"])
    out_df.to_csv(OUT_FILE, index=False)
    print(f"Saved: {OUT_FILE} (rows={len(out_df)})")


if __name__ == "__main__":
    main()
