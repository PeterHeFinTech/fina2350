from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CAR vs sentiment (4 subplots).")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "category_B_car_sentiment.csv"),
        help="Path to input CSV with columns: CAR-1, CAR0, CAR1, CAR2, sentiment",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "car_vs_sentiment_4plots.png"),
        help="Path to save output figure",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    required_cols = ["CAR-1", "CAR0", "CAR1", "CAR2", "sentiment"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    x = df["sentiment"]
    y_cols = ["CAR-1", "CAR0", "CAR1", "CAR2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for ax, y_col in zip(axes, y_cols):
        y = df[y_col]
        ax.scatter(x, y, alpha=0.75, s=25)
        ax.set_title(f"{y_col} vs Sentiment")
        ax.set_xlabel("sentiment")
        ax.set_ylabel(y_col)
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("CAR vs Sentiment", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)

    print(f"Saved figure: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
