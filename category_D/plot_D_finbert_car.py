import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path


BASE = Path(__file__).resolve().parent
IN_FILE = BASE / "D_finbert_car.csv"
OUT_FILE = BASE / "D_finbert_car_plot.png"


def bootstrap_ci_line(x, y, xgrid, n_boot=2000, rng=None):
    rng = np.random.default_rng(rng)
    preds = np.zeros((n_boot, len(xgrid)))
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        xi = x[idx]
        yi = y[idx]
        if np.std(xi) == 0:
            # degenerate, repeat
            m, b = 0.0, yi.mean()
        else:
            m, b = np.polyfit(xi, yi, 1)
        preds[i] = m * xgrid + b
    lower = np.percentile(preds, 2.5, axis=0)
    upper = np.percentile(preds, 97.5, axis=0)
    mean = np.percentile(preds, 50, axis=0)
    return mean, lower, upper


def plot():
    df = pd.read_csv(IN_FILE)
    x = df["sentiment"].values
    cars = ["CAR-1", "CAR0", "CAR1", "CAR2"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    xgrid = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)

    for ax, car in zip(axes, cars):
        y = df[car].values
        # scatter
        ax.scatter(x, y, color="#1f77b4", s=40, alpha=0.9)
        # fit
        if len(x) >= 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            y_pred = m * xgrid + b
            mean, lower, upper = bootstrap_ci_line(x, y, xgrid, n_boot=2000, rng=42)
            ax.plot(xgrid, mean, color="red", lw=2)
            ax.fill_between(xgrid, lower, upper, color="red", alpha=0.2)
        # stats
        try:
            r, p = pearsonr(x, y)
            stat_txt = f"r={r:.3f}, p={p:.3g}"
        except Exception:
            stat_txt = "r=nan"
        ax.set_title(f"{car} vs sentiment — {stat_txt}")
        ax.set_xlabel("FinBERT sentiment (score)")
        ax.set_ylabel(car)
        ax.grid(True, alpha=0.3)
    # set consistent y-limits across all panels
    for ax in axes:
        ax.set_ylim(-0.1, 0.12)
    plt.tight_layout()
    fig.suptitle("Sentiment vs CAR (FinBERT)", y=1.02, fontsize=14)
    plt.savefig(OUT_FILE, bbox_inches="tight", dpi=200)
    print(f"Saved plot: {OUT_FILE}")


if __name__ == "__main__":
    plot()
