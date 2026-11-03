# YES Bank Monthly Closing Price Prediction

This project evaluates one-month-ahead YES Bank closing-price forecasts using the supplied monthly OHLC data. It is an educational time-series example, not investment advice.

## Data and target

The CSV contains 185 monthly records from July 2005 through November 2020. Each training example uses information available at the end of one month to estimate the following month's closing price. The model learns a next-month log return and converts it back to a price using the latest close. Features include recent returns, rolling return averages and volatility, within-month OHLC ranges, and calendar seasonality.

The source contains no news, market indices, or company fundamentals. It cannot identify why the stock moved or support a current-market forecast unless the CSV is updated.

## Evaluation and model choice

The workflow compares a persistence forecast (next month's close equals this month's close) against Ridge, Random Forest, and Histogram Gradient Boosting regression. Hyperparameters are selected using expanding, chronological time-series folds on the first 80% of observations. The last 20% is held out for final evaluation and is not used for tuning.

The selected forecast uses a machine-learning model only when its rolling-fold MAE is at least 5% lower than the persistence baseline. Otherwise, it uses persistence. This guard is intentional: on a small and volatile series, a more complex model should not be presented as better when its cross-validation improvement is marginal. Check `artifacts/metrics.json` for the selected method, cross-validation comparison, holdout MAE/RMSE/MAPE/R², and forecast date. The holdout results may still be worse than persistence; they are an honest out-of-sample check, not a guarantee of future performance.

## Workflow

1. `explore.py` checks data quality and summarizes the monthly series.
2. `train.py` builds return-based features, tunes the candidate models using chronological folds, evaluates the final holdout, and writes metrics, predictions, charts, and a fitted artifact.
3. `predict.py` reads the selected model from that artifact and forecasts the month after the latest CSV row. It handles both the selected ML model and the persistence fallback.

The notebook `yesbank_stock_prediction_analysis.ipynb` provides a guided review. The source deck's description for `Close` is mislabeled; this project treats the `Close` column as closing stock price, consistent with its name and the prediction objective.

## Run

```bash
python -m pip install -r requirements.txt
python explore.py
python train.py --data data/data_YesBank_StockPrices.csv
python predict.py
```

Training writes `artifacts/metrics.json`, `artifacts/test_predictions.csv`, `artifacts/best_model.joblib`, and the holdout comparison and residual charts. The supplied data ends in November 2020, so its next-month forecast is for December 2020. Update the CSV before using the scripts for a newer period.

## Dataset fields

`Date`, `Open`, `High`, `Low`, and `Close` are monthly records. Invalid rows and inconsistent OHLC values are rejected rather than silently used.
