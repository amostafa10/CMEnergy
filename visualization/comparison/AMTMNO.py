import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("./data/macro_eco_factors/AMTMNO.csv")

dataframe['observation_date'] = pd.to_datetime(dataframe['observation_date'] , format='%Y-%m-%d')
# dataframe = dataframe.set_index('observation_date')

print(dataframe)

plt.plot(dataframe['observation_date'], dataframe['AMTMNO'])
# plt.plot(dataframe.index, dataframe['Close'])
plt.xlabel('observation_date')
plt.ylabel('AMTMNO')
plt.title('Graph of Date vs AMTMNO')
plt.xticks(rotation=45)
plt.show()