import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Read from CSV file
csv_path = 'yahoocrudeoildata.csv'
data = pd.read_csv(
    csv_path,
    parse_dates=['Date'], # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date' # Set the date as the index
)

data.sort_index(inplace=True)

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
plt.figure(figsize=(7, 6))
plt.plot(data.index, data['Close_Diff'], label='Differenced Close Price')
plt.title('Differenced Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.legend()
plt.show()

# Autocorrelation and Partial Autocorrelation to determine ARIMA parameters.
plot_acf(data['Close'], lags=40)
plot_pacf(data['Close'], lags=40)
plt.show()



# Graph data
# plt.title('Crude Oil Price vs Time')
# plt.xlabel('Time')
# plt.ylabel('Crude Oil Price')
# plt.plot(data.index, data['Close'])
# plt.show()