# import pandas as pd
# import yfinance as yf
# import csv
# data = yf.download("CL=F", start="2010-01-01", end="2024-12-30")
# print(data)

# df = pd.DataFrame(data)

# # Convert DataFrame to CSV
# df.to_csv('output.csv', index=True)

# # import statsmodels.tsa.statespace.arimax import ARIMAX
# # %matplotlib inline
# # print("Hello world!")

import yfinance as yf
import datetime

currentTime = datetime.datetime.now()
offsetTime = currentTime - datetime.timedelta(days=30*6)

currentTimeString = currentTime.strftime("%Y-%m-%d")
offsetTimeString = offsetTime.strftime("%Y-%m-%d")

print(currentTimeString, offsetTimeString)

result = yf.download("CL=F", start=offsetTimeString, end=currentTimeString)
result.to_csv('test.csv', index=True)