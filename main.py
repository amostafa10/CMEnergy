import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("yahoocrudeoildata.csv", delim_whitespace=True)

data['Date'] = pd.to_datetime(data['Date'])