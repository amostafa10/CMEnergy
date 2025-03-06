import pandas as pd
import matplotlib.pyplot as plt
from auto_ts import auto_timeseries

csv_path = 'data/yahoo_finance_2024.csv'
dataframe = pd.read_csv(csv_path)

model = auto_timeseries(
    score_type='rmse',
    time_interval='D',
    non_seasonal_pdq=None, seasonality=False,
    # seasonal_period=12,
    model_type=['Prophet'],
    verbose=2,
)

model.fit(
    traindata=dataframe,
    ts_column="Date",
    target="Close",
    cv=5,
    sep=","
)

predictions = model.predict(
    testdata = 30,  # can be either a dataframe or an integer standing for the forecast_period,
    model = 'best'  # or any other string that stands for the trained model
)

predictions.plot()