import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Read from CSV file
csv_path = 'yahoocrudeoildata.csv'
data = pd.read_csv(
    csv_path,
    parse_dates=['Date'],  # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date'  # Set the date as the index
)

data.sort_index(inplace=True)

# Graph price over time data
# plt.plot(data.index, data['Close'], label='Close Price')
# plt.title('Crude Oil Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Crude Oil Price')
# plt.legend()
# plt.show()

# Test for stationarity
adf_test_original = adfuller(data['Close'])

adfStatistic_original = adf_test_original[0]
pValue_original = adf_test_original[1]

print('ADF Statistic (original): %f' % adfStatistic_original)
print('p-value (original): %f' % pValue_original)

# A p-value below 0.05 indicates that the data is stationary
if pValue_original < 0.05:
    print('original data is stationary, no differencing needed')
else:
    print('original data is not stationary, differencing needed')


# Difference our data
data['Close_Diff'] = data['Close'].diff()

# Retest our data
adf_test_diff = adfuller(data["Close_Diff"].dropna())
adfStatistic_diff = adf_test_diff[0]
pValue_diff = adf_test_diff[1]

print('ADF Statistic (differenced): %f' % adfStatistic_diff)
print('p-value (differenced): %f' % pValue_diff)

# We are assuming our order of differencing is 1, since that's how much I've had to do in testing.
# Replace with a while-loop to keep differencing and incrementing our order of differencing for a more automated solution.

# Graph differenced data
# plt.figure(figsize=(7, 6))
# plt.plot(data.index, data['Close_Diff'], label='Differenced Close Price')
# plt.title('Differenced Close Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Differenced Close Price')
# plt.legend()
# plt.show()

# Autocorrelation and Partial Autocorrelation to determine ARIMA parameters.
# In terms of our differenced data!
# fig, axes = plt.subplots(1, 2, figsize=(16, 4))

# plot_acf(data['Close_Diff'].dropna(), lags=40, ax=axes[0])
# axes[0].set_title('Autocorrelation Function (ACF)')

# plot_pacf(data['Close_Diff'].dropna(), lags=40, ax=axes[1])
# axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.show()

# See that both the ACF and PACF crosses over at 1, those are our q and p values of our ARIMA model respectively.

# Split our data into testing and training data, we're going to use an 80% / 20% split.
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Fit ARIMA model
model = ARIMA(train['Close'], order=(1, 1, 1))
model_fit = model.fit()

# Test and visualize the results
forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(14, 7))
plt.plot(train.index, train['Close'], label='Train', color='#203147') # Plot the training data
plt.plot(test.index, test['Close'], label='Test', color='#01ef63') # Plot the real testing data
plt.plot(test.index, forecast, label='Forecast', color='orange') # Plot the forecasted data
plt.title('Close Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

print(f'AIC: {model_fit.aic}')
print(f'BIC: {model_fit.bic}')

forecast = forecast[:len(test)]
test_close = test['Close'][:len(forecast)]

rmse = np.sqrt(mean_squared_error(test_close, forecast))
print(f'RMSE: {rmse:.4f}')