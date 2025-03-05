import pandas as pd
import matplotlib.pyplot as plt

# Get our data
csv_path = 'yahoocrudeoildata.csv'
data = pd.read_csv(
    csv_path,
    parse_dates=['Date'],  # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date'  # Set the date as the index
)
data.sort_index(inplace=True)

# Perform a simple rolling average
simple_rolling_average = data['Close'].rolling(50).mean()

# Graph price over time data
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(simple_rolling_average, label='Simple Moving Average')
plt.title('Crude Oil Price Over Time')
plt.xlabel('Date')
plt.ylabel('Crude Oil Price')
plt.legend()
plt.show()