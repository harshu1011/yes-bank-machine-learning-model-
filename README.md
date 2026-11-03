# YES Bank Monthly Closing Price Prediction

This project uses the supplied monthly YES Bank OHLC dataset to predict the **following month's closing price**. It is an educational time-series regression example, not investment advice.

## Why the target is shifted

For each month, the model uses that month's open, high, low, close, recent closing-price history, and calendar fields to predict the next month's close. This means the model does not use the target month's high/low/close as input. All features are available after the current month has ended.

The data contains 185 monthly records from July 2005 through November 2020. It does not contain news, market indices, or company fundamentals, so it cannot isolate the effect of any specific event mentioned in the accompanying project brief.

## Models and evaluation

The training script compares:

- A persistence baseline: next month's close equals this month's close.
- Ridge regression with standardization and time-series cross-validation for alpha selection.
- Random Forest regression with time-series cross-validation for tree settings.

The final 20% of observations are held out chronologically. Model-family selection uses cross-validation on the earlier training period; the holdout is reserved for final reporting. Metrics include MAE, RMSE, MAPE, and R². A model is not considered useful merely because it beats a constant or random baseline; compare it with the persistence baseline and interpret errors in the context of the stock's volatility.

On this dataset's holdout period (December 2017–November 2020), the persistence baseline achieved an MAE of 23.80, compared with 30.58 for Random Forest and 31.74 for Ridge. Although Ridge had the lowest cross-validation MAE in the earlier training period and is saved as the final model, neither trained model beat the simple baseline on the holdout. The large errors reflect how difficult this series is to forecast from OHLC history alone.

## End-to-end workflow

1. `explore.py` checks missing/invalid values, duplicate months, OHLC consistency, and potential IQR outliers. It reports outliers without removing valid market moves. It also writes summary statistics and price, return, and correlation charts.
2. `train.py` creates lag and rolling-price features, tunes Ridge and Random Forest models with `TimeSeriesSplit`, evaluates them against the persistence baseline on the final 20% of months, and saves the final model trained on all available history.
3. `predict.py` loads the saved model and predicts the month after the most recent row in the CSV.

The project also includes `yesbank_stock_prediction_analysis.ipynb` for guided review. The `Close` field is treated as closing price because the source presentation's description for that field is mislabeled. The monthly data does not include headlines or external market drivers, so results cannot establish the impact of a specific news event.

## Run

```bash
python -m pip install -r requirements.txt
python explore.py
python train.py --data data/data_YesBank_StockPrices.csv
python predict.py
```

The scripts write `artifacts/eda_summary.json`, data exploration plots, `artifacts/metrics.json`, `artifacts/test_predictions.csv`, holdout prediction and residual plots, and `artifacts/best_model.joblib`. The saved model is fitted on all available rows after evaluation. Since the source CSV ends in November 2020, its forecast is for December 2020 and should not be interpreted as a current-market forecast. Update the CSV before using `predict.py` for a newer estimate.

## Dataset fields

`Date`, `Open`, `High`, `Low`, and `Close` are monthly records. The source deck contains a mislabeled description for `Close`; this project treats it as the closing stock price, consistent with the column name and project objective.
