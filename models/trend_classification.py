import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'data/yahoo_finance_2024.csv'
data_frame = pd.read_csv(
    csv_path,
    parse_dates=['Date'],  # Parse the dates
    date_format='%Y-%m-%d',
    index_col='Date'  # Set the date as the index
)

# Identify rising/falling
short_term_ma_window = 20
long_term_ma_window = 100

data_frame["short_term_ma"] = data_frame['Close'].rolling(short_term_ma_window).mean()
data_frame["long_term_ma"] = data_frame['Close'].rolling(long_term_ma_window).mean()

# Identify volatility
volatility_window = 20
data_frame['volatility'] = data_frame['Close'].rolling(volatility_window).std()

# Classification function
def classify_trend(row):
    if str(row['volatility']) == 'nan' or str(row['short_term_ma']) == 'nan' or str(row['long_term_ma']) == 'nan':
        return None

    print(row['volatility'])
    if row['volatility'] > row['volatility'].quantile(0.75):  # High volatility
        return 'volatile'
    elif row['short_term_ma'] > row['long_term_ma']:  # Short-term moving average is above long-term MA
        return 'rising'
    elif row['short_term_ma'] < row['long_term_ma']:  # Short-term moving average is below long-term MA
        return 'falling'
    else:
        return 'stable'
    
data_frame['market_trend'] = data_frame.apply(classify_trend, axis=1)

# Plot the price, moving averages, and market trends
plt.figure(figsize=(10, 6))
plt.plot(data_frame['Close'], label='Crude Oil Price', color='blue', alpha=0.5)
plt.plot(data_frame['short_term_ma'], label='Short Term MA', color='orange')
plt.plot(data_frame['long_term_ma'], label='Long Term MA', color='green')

for start, end in zip(data_frame.index[data_frame['market_trend'] != data_frame['market_trend'].shift()].values, data_frame.index[1:][data_frame['market_trend'] != data_frame['market_trend'].shift()].values):
    plt.axvspan(start, end, color='gray', alpha=0.3)

plt.legend()
plt.title('Crude Oil Price with Market Trend Classification')
plt.show()