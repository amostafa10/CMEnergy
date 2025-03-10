import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("./data/macro_eco_factors/HOUST.csv")

dataframe['observation_date'] = pd.to_datetime(dataframe['observation_date'] , format='%Y-%m-%d')
# dataframe = dataframe.set_index('observation_date')

print(dataframe)

plt.plot(dataframe['observation_date'], dataframe['HOUST'])
# plt.plot(dataframe.index, dataframe['Close'])
plt.xlabel('observation_date')
plt.ylabel('HOUST')
plt.title('Graph of Date vs HOUST')
# plt.xticks(rotation=45)
plt.show()