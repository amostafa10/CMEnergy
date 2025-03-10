import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("./data/macro_eco_factors/INDPRO.csv")

dataframe['observation_date'] = pd.to_datetime(dataframe['observation_date'] , format='%Y-%m-%d')
# dataframe = dataframe.set_index('observation_date')

print(dataframe)

plt.plot(dataframe['observation_date'], dataframe['INDPRO'])
# plt.plot(dataframe.index, dataframe['Close'])
plt.xlabel('observation_date')
plt.ylabel('INDPRO')
plt.title('Graph of Date vs INDPRO')
# plt.xticks(rotation=45)
plt.show()