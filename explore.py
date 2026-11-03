"""Create data-quality and exploratory analysis outputs for the YES Bank CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train import DEFAULT_DATA, load_data


ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PRICE_COLUMNS = ["open", "high", "low", "close"]


def quality_report(path: Path, clean: pd.DataFrame) -> dict:
    source = pd.read_csv(path)
    source.columns = [str(column).strip().lower() for column in source.columns]
    date_values = pd.to_datetime(source.get("date", pd.Series(dtype=str)).astype(str).str.strip(), format="%b-%y", errors="coerce")
    numeric = source.reindex(columns=PRICE_COLUMNS).apply(pd.to_numeric, errors="coerce")
    invalid_date_rows = int(date_values.isna().sum())
    invalid_numeric_rows = int(numeric.isna().any(axis=1).sum()) if not numeric.empty else len(source)
    parsed_dates = pd.DataFrame({"date": date_values}).dropna()
    duplicate_months = int(parsed_dates["date"].duplicated().sum())
    outliers = {}
    for column in PRICE_COLUMNS:
        q1, q3 = clean[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[column] = int(((clean[column] < low_fence) | (clean[column] > high_fence)).sum())

    return {
        "source_rows": int(len(source)),
        "clean_rows": int(len(clean)),
        "rows_removed": int(len(source) - len(clean)),
        "missing_by_column": {str(k): int(v) for k, v in source.isna().sum().items()},
        "invalid_date_rows": invalid_date_rows,
        "rows_with_invalid_numeric_values": invalid_numeric_rows,
        "duplicate_months": duplicate_months,
        "outlier_flags_iqr": outliers,
        "outlier_policy": "Report potential IQR outliers but keep valid observations, since large price moves may be genuine market events.",
        "ohlc_consistency_violations": {
            "high_below_open_or_close": int(((clean["high"] < clean[["open", "close"]].max(axis=1))).sum()),
            "low_above_open_or_close": int(((clean["low"] > clean[["open", "close"]].min(axis=1))).sum()),
            "high_below_low": int((clean["high"] < clean["low"]).sum()),
        },
    }


def make_plots(frame: pd.DataFrame) -> None:
    ARTIFACTS.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(frame["date"], frame["close"], color="#17324d", linewidth=1.8)
    ax.set(title="YES Bank monthly closing price", xlabel="Month", ylabel="Closing price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "price_history.png", dpi=160)
    plt.close(fig)

    returns = frame.set_index("date")["close"].pct_change().dropna().mul(100)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = np.where(returns >= 0, "#2a8c68", "#d9534f")
    ax.bar(returns.index, returns, width=22, color=colors, alpha=0.8)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set(title="Month-to-month change in closing price", xlabel="Month", ylabel="Change (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "monthly_returns.png", dpi=160)
    plt.close(fig)

    corr = frame[PRICE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(PRICE_COLUMNS)), [x.title() for x in PRICE_COLUMNS])
    ax.set_yticks(range(len(PRICE_COLUMNS)), [x.title() for x in PRICE_COLUMNS])
    for row in range(len(PRICE_COLUMNS)):
        for col in range(len(PRICE_COLUMNS)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", color="black")
    ax.set_title("Correlation among monthly OHLC fields")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "ohlc_correlation.png", dpi=160)
    plt.close(fig)


def analyze(path: Path) -> dict:
    clean = load_data(path)
    report = quality_report(path, clean)
    report["date_range"] = {
        "first_month": clean["date"].min().strftime("%Y-%m"),
        "last_month": clean["date"].max().strftime("%Y-%m"),
    }
    report["descriptive_statistics"] = {
        k: {str(stat): float(value) for stat, value in values.items()}
        for k, values in clean[PRICE_COLUMNS].describe().to_dict().items()
    }
    report["close_monthly_return_percent"] = {
        "mean": float(clean["close"].pct_change().mean() * 100),
        "median": float(clean["close"].pct_change().median() * 100),
        "minimum": float(clean["close"].pct_change().min() * 100),
        "maximum": float(clean["close"].pct_change().max() * 100),
    }
    report["ohlc_correlation"] = clean[PRICE_COLUMNS].corr().round(4).to_dict()
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_plots(clean)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the monthly OHLC CSV")
    args = parser.parse_args()
    print(json.dumps(analyze(args.data), indent=2))


if __name__ == "__main__":
    main()
