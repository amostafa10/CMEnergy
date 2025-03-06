import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv("./data/BOGMBASE.csv")

dataframe['observation_date'] = pd.to_datetime(dataframe['observation_date'] , format='%Y-%d-%m')
# dataframe = dataframe.set_index('observation_date')

print(dataframe)

plt.plot(dataframe['observation_date'], dataframe['BOGMBASE'])
# plt.plot(dataframe.index, dataframe['Close'])
plt.xlabel('observation_date')
plt.ylabel('BOGMBASE')
plt.title('Graph of Date vs BOGMBASE')
# plt.xticks(rotation=45)
plt.show()