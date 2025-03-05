import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import pmdarima as pmd

# Read from CSV file
csv_path = 'yahoocrudeoildata.csv'
data = pd.read_csv(
    csv_path,
    parse_dates=['Date'],  # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date'  # Set the date as the index
)
data.sort_index(inplace=True)

model = pmd.auto_arima(data['Close'], start_p=1, start_q=1, test='adf', m=12, seasonal=True, trace=1)

# https://www.geeksforgeeks.org/sarima-seasonal-autoregressive-integrated-moving-average/
# https://medium.com/@tirthamutha/time-series-forecasting-using-sarima-in-python-8b75cd3366f2
# https://neptune.ai/blog/arima-sarima-real-world-time-series-forecasting-guide
# https://www.machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/