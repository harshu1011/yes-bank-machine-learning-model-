"""Train and evaluate one-month-ahead YES Bank closing-price regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data_YesBank_StockPrices.csv"
ARTIFACTS = ROOT / "artifacts"
FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_mean_3",
    "close_mean_6",
    "month_sin",
    "month_cos",
    "year",
]


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    frame["date"] = pd.to_datetime(frame["date"].astype(str).str.strip(), format="%b-%y", errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close"])
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(frame) < 60:
        raise ValueError(f"Expected at least 60 valid monthly rows; found {len(frame)}")
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[["date", "open", "high", "low", "close"]].copy()
    for lag in (1, 2, 3):
        features[f"close_lag_{lag}"] = frame["close"].shift(lag)
    features["close_mean_3"] = frame["close"].rolling(3).mean()
    features["close_mean_6"] = frame["close"].rolling(6).mean()
    month = frame["date"].dt.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    features["year"] = frame["date"].dt.year
    return features[FEATURE_COLUMNS]


def make_supervised(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = make_features(frame)
    # Row t predicts close at t+1. The current month close is allowed because
    # predictions are made after month t has completed.
    target = frame["close"].shift(-1).rename("target_next_close")
    target_date = frame["date"].shift(-1).rename("target_date")
    valid = features.notna().all(axis=1) & target.notna() & target_date.notna()
    return (
        features.loc[valid].reset_index(drop=True),
        target.loc[valid].reset_index(drop=True),
        target_date.loc[valid].reset_index(drop=True),
    )


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    nonzero = np.asarray(y_true) != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(np.mean(np.abs((np.asarray(y_true)[nonzero] - y_pred[nonzero]) / np.asarray(y_true)[nonzero])) * 100),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train(data_path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = load_data(data_path)
    X, y, dates = make_supervised(raw)
    split_at = int(len(X) * 0.8)
    if split_at <= 30 or len(X) - split_at < 10:
        raise ValueError("Not enough rows for an 80/20 chronological train/test split")

    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    test_dates = dates.iloc[split_at:]
    cv = TimeSeriesSplit(n_splits=4)

    ridge_search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
        {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
    ).fit(X_train, y_train)

    forest_search = GridSearchCV(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        {
            "max_depth": [2, 4, None],
            "min_samples_leaf": [2, 4],
            "max_features": [0.7, 1.0],
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
    ).fit(X_train, y_train)

    searches = {"ridge": ridge_search, "random_forest": forest_search}
    models = {name: search.best_estimator_ for name, search in searches.items()}
    test_predictions = {name: model.predict(X_test) for name, model in models.items()}
    baseline_predictions = X_test["close"].to_numpy()
    cv_mae = {name: float(-search.best_score_) for name, search in searches.items()}
    selected_name = min(cv_mae, key=cv_mae.get)
    selected_model = models[selected_name]
    ridge_coefficients = models["ridge"].named_steps["model"].coef_
    forest_importance = models["random_forest"].feature_importances_

    results = {
        "dataset": {
            "rows": int(len(raw)),
            "first_month": raw["date"].min().strftime("%Y-%m"),
            "last_month": raw["date"].max().strftime("%Y-%m"),
            "supervised_rows": int(len(X)),
            "target": "following_month_close",
        },
        "split": {
            "method": "chronological 80/20",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_first_target_month": pd.Timestamp(test_dates.iloc[0]).strftime("%Y-%m"),
            "test_last_target_month": pd.Timestamp(test_dates.iloc[-1]).strftime("%Y-%m"),
        },
        "selection": {
            "method": "lowest training-period TimeSeriesSplit cross-validation MAE",
            "selected_model": selected_name,
            "cross_validation_mae": cv_mae,
            "best_parameters": {name: search.best_params_ for name, search in searches.items()},
        },
        "feature_insights": {
            "ridge_standardized_coefficients": {
                name: float(value)
                for name, value in sorted(zip(FEATURE_COLUMNS, ridge_coefficients), key=lambda item: abs(item[1]), reverse=True)
            },
            "random_forest_importance": {
                name: float(value)
                for name, value in sorted(zip(FEATURE_COLUMNS, forest_importance), key=lambda item: item[1], reverse=True)
            },
            "interpretation_note": "Feature associations are not causal. OHLC values within a month are highly related and the dataset contains no news or market-wide variables.",
        },
        "holdout_metrics": {
            "persistence_baseline": score(y_test, baseline_predictions),
            **{name: score(y_test, preds) for name, preds in test_predictions.items()},
        },
    }

    ARTIFACTS.mkdir(exist_ok=True)
    output = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates).dt.strftime("%Y-%m").to_numpy(),
            "actual_close": y_test.to_numpy(),
            "persistence_prediction": baseline_predictions,
            **{f"{name}_prediction": preds for name, preds in test_predictions.items()},
        }
    )
    output.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(pd.to_datetime(test_dates), y_test, label="Actual", color="#202a44", linewidth=2)
    ax.plot(pd.to_datetime(test_dates), baseline_predictions, label="Persistence baseline", color="#8b98ad", linestyle="--")
    ax.plot(pd.to_datetime(test_dates), test_predictions[selected_name], label=f"Selected: {selected_name}", color="#e54b4b")
    ax.set(title="YES Bank next-month closing price: holdout period", xlabel="Target month", ylabel="Closing price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    residuals = y_test.to_numpy() - test_predictions[selected_name]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.plot(pd.to_datetime(test_dates), residuals, marker="o", color="#7a4eab", linewidth=1.3)
    ax.set(title=f"Holdout prediction errors: {selected_name}", xlabel="Target month", ylabel="Actual − predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "residuals.png", dpi=160)
    plt.close(fig)

    model_names = ["persistence_baseline", "ridge", "random_forest"]
    mae_values = [results["holdout_metrics"][name]["mae"] for name in model_names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([name.replace("_", " ").title() for name in model_names], mae_values, color=["#8492a6", "#e54b4b", "#3b82a0"])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(title="Holdout MAE comparison (lower is better)", ylabel="Mean absolute error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "holdout_mae_comparison.png", dpi=160)
    plt.close(fig)

    # Refit the CV-selected model on all historical supervised rows for use in
    # the next-month forecast; holdout metrics above remain from the untouched test set.
    final_search = searches[selected_name]
    final_model = final_search.best_estimator_.set_params(**final_search.best_params_)
    final_model.fit(X, y)
    dump(final_model, ARTIFACTS / "best_model.joblib")
    next_prediction = float(final_model.predict(make_features(raw).iloc[[-1]])[0])
    results["next_month_forecast"] = {
        "target_month": (raw["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "predicted_close": next_prediction,
        "training_data_through": raw["date"].max().strftime("%Y-%m"),
        "note": "Historical forecast only; dataset ends in November 2020 and this is not a current-market estimate.",
    }

    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the monthly OHLC CSV")
    args = parser.parse_args()
    results = train(args.data)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
