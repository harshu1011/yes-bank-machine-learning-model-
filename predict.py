"""Load the trained model and forecast the month after the latest CSV row."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load

from train import ARTIFACTS, DEFAULT_DATA, make_features, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the monthly OHLC CSV")
    parser.add_argument("--model", type=Path, default=ARTIFACTS / "best_model.joblib", help="Path to a fitted joblib model")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model file not found: {args.model}. Run train.py first.")

    frame = load_data(args.data)
    latest_features = make_features(frame).iloc[[-1]]
    if latest_features.isna().any(axis=None):
        raise SystemExit("Not enough monthly history to build lag features for a forecast.")
    forecast = float(load(args.model).predict(latest_features)[0])
    target_month = (frame["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    print(f"Training data through: {frame['date'].max():%Y-%m}")
    print(f"Forecast target month: {target_month}")
    print(f"Predicted closing price: {forecast:.2f}")
    print("This is a historical-data forecast. It is only useful for a current estimate if the CSV is updated first.")


if __name__ == "__main__":
    main()
