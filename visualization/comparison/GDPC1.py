import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("./data/macro_eco_factors/GDPC1.csv")

dataframe['observation_date'] = pd.to_datetime(dataframe['observation_date'] , format='%Y-%m-%d')
# dataframe = dataframe.set_index('observation_date')

print(dataframe)

plt.plot(dataframe['observation_date'], dataframe['GDPC1'])
# plt.plot(dataframe.index, dataframe['Close'])
plt.xlabel('observation_date')
plt.ylabel('GDPC1')
plt.title('Graph of Date vs GDPC1')
# plt.xticks(rotation=45)
plt.show()