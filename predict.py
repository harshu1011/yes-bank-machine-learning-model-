"""Load the trained model and forecast the month after the latest CSV row."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    artifact = load(args.model)
    latest_features = make_features(frame).iloc[[-1]]
    if latest_features.isna().any(axis=None):
        raise SystemExit("Not enough monthly history to build lag features for a forecast.")
    method = artifact["selected_model"]
    if method == "persistence_baseline":
        forecast = float(frame["close"].iloc[-1])
    else:
        latest_features = latest_features[artifact["feature_columns"]]
        predicted_log_return = float(artifact["estimator"].predict(latest_features)[0])
        forecast = float(frame["close"].iloc[-1] * np.exp(predicted_log_return))
    target_month = (frame["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    print(f"Training data through: {frame['date'].max():%Y-%m}")
    print(f"Forecast target month: {target_month}")
    print(f"Predicted closing price: {forecast:.2f}")
    print(f"Forecast method: {method}")
    print("This is a historical-data forecast. It is only useful for a current estimate if the CSV is updated first.")


if __name__ == "__main__":
    main()
