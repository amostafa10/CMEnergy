import pandas as pd
import conda
import yfinance as yf
import statsmodels.tsa.statespace.arimax import ARIMAX
%matplotlib inline
data = yf.download("CL=F", start="2010-01-01", end="2020-01-01")
print(data.head())
print(data)
print("Hello world!")