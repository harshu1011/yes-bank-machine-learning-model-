"""Train, tune, and evaluate one-month-ahead YES Bank price forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "data_YesBank_StockPrices.csv"
ARTIFACTS = ROOT / "artifacts"
FEATURE_COLUMNS = [
    "open_gap",
    "high_range",
    "low_range",
    "return_1m",
    "return_2m",
    "return_3m",
    "return_mean_3m",
    "return_mean_6m",
    "return_volatility_3m",
    "return_volatility_6m",
    "month_sin",
    "month_cos",
]
MIN_CV_IMPROVEMENT = 0.05
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    """Load, validate, sort, and clean monthly OHLC data."""
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
    invalid_ohlc = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    if invalid_ohlc.any():
        raise ValueError(f"Found {int(invalid_ohlc.sum())} rows with inconsistent OHLC values")
    if len(frame) < 60:
        raise ValueError(f"Expected at least 60 valid monthly rows; found {len(frame)}")
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build end-of-month features for a next-month log-return model."""
    close = frame["close"]
    monthly_return = close.pct_change()
    month = frame["date"].dt.month
    features = pd.DataFrame(
        {
            "open_gap": frame["open"] / close - 1,
            "high_range": frame["high"] / close - 1,
            "low_range": frame["low"] / close - 1,
            "return_1m": close.pct_change(1),
            "return_2m": close.pct_change(2),
            "return_3m": close.pct_change(3),
            "return_mean_3m": monthly_return.rolling(3).mean(),
            "return_mean_6m": monthly_return.rolling(6).mean(),
            "return_volatility_3m": monthly_return.rolling(3).std(),
            "return_volatility_6m": monthly_return.rolling(6).std(),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=frame.index,
    )
    return features[FEATURE_COLUMNS]


def make_supervised(frame: pd.DataFrame):
    """Return features, next-month log returns, dates, anchor prices, and targets."""
    features = make_features(frame)
    target_return = np.log(frame["close"].shift(-1) / frame["close"])
    target_close = frame["close"].shift(-1)
    target_date = frame["date"].shift(-1)
    anchor_close = frame["close"]
    valid = features.notna().all(axis=1) & target_return.notna() & target_close.notna()
    return (
        features.loc[valid].reset_index(drop=True),
        target_return.loc[valid].reset_index(drop=True),
        target_date.loc[valid].reset_index(drop=True),
        anchor_close.loc[valid].reset_index(drop=True),
        target_close.loc[valid].reset_index(drop=True),
    )


def score(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    nonzero = actual != 0
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_percent": float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100),
        "r2": float(r2_score(actual, predicted)),
    }


def model_grids() -> dict[str, tuple[object, dict]]:
    """Return modest candidate grids suitable for this small dataset."""
    return {
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [1.0, 10.0, 100.0]},
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=1),
            {"max_depth": [2, 4, None], "min_samples_leaf": [2, 4], "max_features": [0.7, 1.0]},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            {"max_iter": [50, 100], "learning_rate": [0.03, 0.1], "max_leaf_nodes": [5, 10]},
        ),
    }


def tune_on_rolling_folds(X: pd.DataFrame, returns: pd.Series, anchors: pd.Series, targets: pd.Series):
    """Tune each family by rolling-fold MAE in the original price units."""
    splitter = TimeSeriesSplit(n_splits=4)
    fold_indexes = list(splitter.split(X))
    baseline_errors = []
    for _, valid_index in fold_indexes:
        baseline_errors.extend(np.abs(targets.iloc[valid_index].to_numpy() - anchors.iloc[valid_index].to_numpy()))
    baseline_cv_mae = float(np.mean(baseline_errors))

    tuned = {}
    for name, (estimator, grid) in model_grids().items():
        candidates = []
        for params in ParameterGrid(grid):
            fold_errors = []
            for train_index, valid_index in fold_indexes:
                fitted = clone(estimator).set_params(**params)
                fitted.fit(X.iloc[train_index], returns.iloc[train_index])
                predicted_return = fitted.predict(X.iloc[valid_index])
                predicted_close = anchors.iloc[valid_index].to_numpy() * np.exp(predicted_return)
                fold_errors.extend(np.abs(targets.iloc[valid_index].to_numpy() - predicted_close))
            candidates.append({"params": params, "cv_mae": float(np.mean(fold_errors))})
        tuned[name] = min(candidates, key=lambda candidate: candidate["cv_mae"])
    return tuned, baseline_cv_mae


def _save_plots(dates, actual, baseline, candidate, selected, selected_name, results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ARTIFACTS.mkdir(exist_ok=True)
    date_values = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(date_values, actual, label="Actual", color="#202a44", linewidth=2)
    ax.plot(date_values, baseline, label="Persistence baseline", color="#8b98ad", linestyle="--")
    ax.plot(date_values, candidate, label=results["selection"]["best_ml_candidate"], color="#e54b4b")
    ax.plot(date_values, selected, label=f"Selected forecast: {selected_name}", color="#2a8c68", alpha=0.8)
    ax.set(title="YES Bank next-month closing price: chronological holdout", xlabel="Target month", ylabel="Closing price")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.plot(date_values, actual - selected, marker="o", color="#7a4eab", linewidth=1.3)
    ax.set(title=f"Holdout errors for selected forecast: {selected_name}", xlabel="Target month", ylabel="Actual − predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "residuals.png", dpi=160)
    plt.close(fig)

    names = list(results["holdout_metrics"])
    values = [results["holdout_metrics"][name]["mae"] for name in names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar([name.replace("_", " ").title() for name in names], values, color=["#8492a6"] + ["#3b82a0"] * (len(names) - 1))
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.set(title="Holdout MAE comparison (lower is better)", ylabel="Mean absolute error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "holdout_mae_comparison.png", dpi=160)
    plt.close(fig)


def train(data_path: Path) -> dict:
    raw = load_data(data_path)
    X, returns, dates, anchors, targets = make_supervised(raw)
    split_at = int(len(X) * 0.8)
    if split_at <= 40 or len(X) - split_at < 10:
        raise ValueError("Not enough rows for an 80/20 chronological train/test split")

    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    return_train, return_test = returns.iloc[:split_at], returns.iloc[split_at:]
    train_anchors, test_anchors = anchors.iloc[:split_at], anchors.iloc[split_at:]
    y_train, y_test = targets.iloc[:split_at], targets.iloc[split_at:]
    test_dates = dates.iloc[split_at:]

    tuned, baseline_cv_mae = tune_on_rolling_folds(X_train, return_train, train_anchors, y_train)
    best_name = min(tuned, key=lambda name: tuned[name]["cv_mae"])
    best_cv_mae = tuned[best_name]["cv_mae"]
    cv_improvement = (baseline_cv_mae - best_cv_mae) / baseline_cv_mae if baseline_cv_mae else 0.0
    selected_name = best_name if cv_improvement >= MIN_CV_IMPROVEMENT else "persistence_baseline"

    fitted_models = {}
    test_predictions = {}
    final_candidates = {}
    for name, (estimator, _) in model_grids().items():
        params = tuned[name]["params"]
        fitted = clone(estimator).set_params(**params).fit(X_train, return_train)
        fitted_models[name] = fitted
        test_predictions[name] = test_anchors.to_numpy() * np.exp(fitted.predict(X_test))
        final_candidates[name] = clone(estimator).set_params(**params).fit(X, returns)

    persistence_predictions = test_anchors.to_numpy()
    selected_predictions = persistence_predictions if selected_name == "persistence_baseline" else test_predictions[selected_name]

    results = {
        "dataset": {
            "rows": int(len(raw)),
            "first_month": raw["date"].min().strftime("%Y-%m"),
            "last_month": raw["date"].max().strftime("%Y-%m"),
            "supervised_rows": int(len(X)),
            "target": "following_month_log_return_converted_to_close",
        },
        "split": {
            "method": "chronological 80/20 holdout",
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_first_target_month": pd.Timestamp(test_dates.iloc[0]).strftime("%Y-%m"),
            "test_last_target_month": pd.Timestamp(test_dates.iloc[-1]).strftime("%Y-%m"),
        },
        "selection": {
            "method": "rolling-fold price MAE; use ML only when CV improvement over persistence is at least 5%",
            "selected_model": selected_name,
            "best_ml_candidate": best_name,
            "cv_persistence_mae": baseline_cv_mae,
            "best_ml_cv_mae": best_cv_mae,
            "cv_improvement_fraction": float(cv_improvement),
            "minimum_improvement_required": MIN_CV_IMPROVEMENT,
            "tuned_models": tuned,
        },
        "holdout_metrics": {
            "persistence_baseline": score(y_test.to_numpy(), persistence_predictions),
            **{name: score(y_test.to_numpy(), values) for name, values in test_predictions.items()},
            "selected_forecast": score(y_test.to_numpy(), selected_predictions),
        },
        "data_notes": [
            "The source has monthly OHLC data only; it has no news, fundamentals, or broad-market features.",
            "A strong result against the training folds is not proof of future performance; the chronological holdout remains the final check.",
        ],
    }

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates).dt.strftime("%Y-%m").to_numpy(),
            "actual_close": y_test.to_numpy(),
            "persistence_prediction": persistence_predictions,
            **{f"{name}_prediction": values for name, values in test_predictions.items()},
            "selected_prediction": selected_predictions,
        }
    )
    ARTIFACTS.mkdir(exist_ok=True)
    output.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    _save_plots(test_dates, y_test.to_numpy(), persistence_predictions, test_predictions[best_name], selected_predictions, selected_name, results)

    # Fit candidate ML model(s) on all observations for reproducibility. The deployed
    # forecast still uses persistence when cross-validation does not show a material gain.
    artifact = {
        "selected_model": selected_name,
        "best_ml_candidate": best_name,
        "estimator": final_candidates[best_name],
        "feature_columns": FEATURE_COLUMNS,
        "minimum_improvement_required": MIN_CV_IMPROVEMENT,
    }
    dump(artifact, ARTIFACTS / "best_model.joblib")
    latest_features = make_features(raw).iloc[[-1]]
    candidate_forecast = float(raw["close"].iloc[-1] * np.exp(artifact["estimator"].predict(latest_features)[0]))
    forecast = candidate_forecast if selected_name != "persistence_baseline" else float(raw["close"].iloc[-1])
    results["next_month_forecast"] = {
        "target_month": (raw["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "predicted_close": forecast,
        "forecast_method": selected_name,
        "best_ml_candidate_prediction": candidate_forecast,
        "training_data_through": raw["date"].max().strftime("%Y-%m"),
        "note": "The data ends in November 2020. This is a historical-data forecast, not a current-market estimate.",
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the monthly OHLC CSV")
    args = parser.parse_args()
    print(json.dumps(train(args.data), indent=2))


if __name__ == "__main__":
    main()
