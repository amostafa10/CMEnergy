import pandas as pd
import conda
import yfinance as yf
import csv
data = yf.download("CL=F", start="2010-01-01", end="2024-12-30")
print(data)

df = pd.DataFrame(data)

# Convert DataFrame to CSV
df.to_csv('output.csv', index=True)

# import statsmodels.tsa.statespace.arimax import ARIMAX
# %matplotlib inline
# print("Hello world!")