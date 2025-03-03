import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("yahoocrudeoildata.csv")

dataframe['Date'] = pd.to_datetime(dataframe['Date'])
dataframe = dataframe.set_index('Date')

print(dataframe)

plt.title('Crude Oil Price vs Time')
plt.xlabel('Time')
plt.ylabel('Crude Oil Price')
plt.plot(dataframe.index, dataframe['Close'])
plt.show()
