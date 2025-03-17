import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
import os

scaler = MinMaxScaler(feature_range=(0, 1))

finance_data_path = 'data/yahoo_finance_2010.csv'
finance_dataframe = pd.read_csv(
    finance_data_path,
    parse_dates=['Date'],
    date_format='%Y-%m-%d',
    index_col='Date'
)

finance_dataframe['Close'] = scaler.fit_transform(finance_dataframe[['Close']])
finance_dataframe['High'] = scaler.fit_transform(finance_dataframe[['High']])
finance_dataframe['Low'] = scaler.fit_transform(finance_dataframe[['Low']])
finance_dataframe['Open'] = scaler.fit_transform(finance_dataframe[['Open']])
finance_dataframe['Volume'] = scaler.fit_transform(finance_dataframe[['Volume']])
# print(finance_dataframe['Close'])

# plt.figure(figsize=(12, 6))
# plt.plot(finance_dataframe.index, finance_dataframe['Close'], label='Close Price (Target)')

macro_eco_factors_path = 'data/macro_eco_factors'
for macro_eco_factor_csv in os.scandir(macro_eco_factors_path):
    if macro_eco_factor_csv.is_file():
        macro_eco_factor_dataframe = pd.read_csv(
            macro_eco_factor_csv,
            parse_dates=['observation_date'],
            date_format='%Y-%m-%d',
            index_col='observation_date'
        )

        factor_name = macro_eco_factor_csv.name.removesuffix('.csv')
        # print(factor_name)
        macro_eco_factor_dataframe = macro_eco_factor_dataframe.loc[(macro_eco_factor_dataframe.index >= '2010-1-4')]

        macro_eco_factor_dataframe[factor_name] = scaler.fit_transform(macro_eco_factor_dataframe[[factor_name]])
        finance_dataframe = finance_dataframe.join(macro_eco_factor_dataframe, how='outer')
        # plt.plot(macro_eco_factor_dataframe.index, macro_eco_factor_dataframe[factor_name], label=factor_name)

finance_dataframe.fillna(finance_dataframe.mean(), inplace=True)

# sb.heatmap(finance_dataframe.corr(), cmap="RdBu_r")
# sb.pairplot(finance_dataframe)
# plt.show()

corr_matrix = finance_dataframe.corr()
corr_matrix['Close'].sort_values(ascending=False)
print(corr_matrix)

# plt.title(f'Combined Graph of Close and Macro Economic Factors')
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()