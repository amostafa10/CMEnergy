import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

data_file_path = 'data/yahoo_finance_2024.csv'
dataframe = pd.read_csv(
    data_file_path,
    parse_dates=['Date'],  # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date'  # Set the date as the index
)

dataframe.sort_index(inplace=True)

# Split training and testing data
test_size = 0.2

split_idx = int(len(dataframe) * (1 - test_size))
train_data = dataframe.iloc[:split_idx].copy()
test_data = dataframe.iloc[split_idx:].copy()

# Set exogenous variables and target variable
exog_cols = ["Open", "High", "Low", "Volume"]
target_col = "Close"

def grid_search(p_range=(0, 2), d_range=(0, 2), q_range=(0, 2), 
                    P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s=20):
        """
        Perform grid search to find optimal SARIMA parameters.
        
        Parameters:
        - p_range, d_range, q_range: Ranges for ARIMA parameters
        - P_range, D_range, Q_range: Ranges for seasonal ARIMA parameters
        - s: Seasonal period
        
        Returns:
        - Dictionary of best parameters
        """
        print("Starting grid search for optimal parameters...")
        print("This may take some time...")
        
        best_aic = float('inf')
        best_params = None
        
        # Create parameter combinations
        p_params = range(p_range[0], p_range[1] + 1)
        d_params = range(d_range[0], d_range[1] + 1)
        q_params = range(q_range[0], q_range[1] + 1)
        P_params = range(P_range[0], P_range[1] + 1)
        D_params = range(D_range[0], D_range[1] + 1)
        Q_params = range(Q_range[0], Q_range[1] + 1)
        
        # Get exogenous variables if specified
        exog_train = train_data[exog_cols]
        
        # Grid search
        for params in itertools.product(p_params, d_params, q_params, P_params, D_params, Q_params):
            p, d, q, P, D, Q = params
            
            # Skip non-invertible models
            if p + d + q + P + D + Q == 0:
                continue
                
            try:
                model = SARIMAX(
                    train_data[target_col],
                    exog=exog_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit(disp=False)
                aic = results.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = {
                        'order': (p, d, q),
                        'seasonal_order': (P, D, Q, s),
                        'aic': aic
                    }
                    print(f"New best parameters: SARIMA{(p,d,q)}x{(P,D,Q,s)} - AIC: {aic}")
            
            except Exception as e:
                continue

        print(f"Best parameters: SARIMA{best_params['order']}x{best_params['seasonal_order']}")
        print(f"Best AIC: {best_params['aic']}")
        
        return best_params


# best_params = grid_search()
# print(best_params)
# Best parameters: SARIMA(0, 0, 0)x(0, 1, 2, 20)
# Best AIC: 14.0


# Train model
order = (0, 0, 0)
seasonal_order = (0, 1, 2, 20) # assuming 20 as in 20 trading days

exog_train = train_data[exog_cols]

model = SARIMAX(
    train_data[target_col],
    exog=exog_train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Model trained
results = model.fit(disp=False)
print("Model training complete.")
print(results.summary())

# Test model
exog_test = test_data[exog_cols]

start = len(train_data)
end = start + len(test_data) - 1

# Make predictions
# predictions = results.predict(
#     start=start,
#     end=end,
#     exog=exog_test
# )

# predictions.index = test_data.index

# # Evaluate metrics
# rmse = np.sqrt(mean_squared_error(test_data[target_col], predictions))
# mae = mean_absolute_error(test_data[target_col], predictions)
# mape = np.mean(np.abs((test_data[target_col] - predictions) / test_data[target_col])) * 100

# print("Model Evaluation Metrics:")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"MAPE: {mape:.4f}%")

# # Plot actual vs predicted
# plt.figure(figsize=(12, 6))
# plt.plot(train_data.index, train_data[target_col], label = "Historical")
# plt.plot(test_data.index, test_data[target_col], label='Actual')
# plt.plot(test_data.index, predictions, label='Predicted', color='red')
# plt.title('SARIMAX Forecast vs Actual')
# plt.xlabel('Date')
# plt.ylabel(target_col)
# plt.legend()
# plt.grid(True)
# plt.show()

# Start predicting the future
steps = 30 # 30 days

# Generate future exogenous variables
last_date = dataframe.index[-1]
frequency = pd.infer_freq(dataframe.index)

if frequency is None:
    date_diffs = dataframe.index[1:] - dataframe.index[:-1]
    most_common_diff = pd.Series(date_diffs).value_counts().index[0]
    frequency = most_common_diff

print(frequency)

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=steps,
    freq=frequency
)

exog_future = pd.DataFrame(index=future_dates)

for exog_var_col in exog_cols:
    model = ARIMA(dataframe[exog_var_col], order=(1, 1, 1))
    model_fit = model.fit()

    exog_forecast = model_fit.forecast(steps=steps)
    print(exog_forecast)
    exog_future[exog_var_col] = exog_forecast

print(exog_future)