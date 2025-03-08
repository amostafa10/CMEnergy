import requests
import pandas as pd
from datetime import datetime

api_key = "ef84caaeb16b21f8c9dbc2840a8cc21f"
symbols = ["AMTMNO", "BOGMBASE", "CPIAUCSL", "GDPC1", "HOUST", "INDPRO", "NCBCMDPMVCE", "PCEC96", "PPIACO", "UNRATE"]

def fetch_and_interpolate(symbol, api_key):
    # FRED API URL to get the monthly observations for each symbol, limiting data to after 2010
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={symbol}&api_key={api_key}&file_type=json&start_date=2010-01-01"
    
    # Make the request to the FRED API
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        observations = data.get('observations', [])
        
        if observations:
            # Prepare data for interpolation (just date and value)
            dates = []
            values = []
            
            for observation in observations:
                date = observation['date']
                value = observation['value']
                
                # Only append valid data (i.e., value is a valid float)
                try:
                    float_value = float(value)  # Try converting the value to float
                    dates.append(datetime.strptime(date, '%Y-%m-%d'))
                    values.append(float_value)
                except ValueError:
                    print(f"Skipping invalid data for {symbol} on {date}: {value}")
                    continue  # Skip invalid data and move to the next observation
            
            # Convert date strings to datetime objects
            if dates:  # Only proceed if there are valid dates and values
                # DataFrame to hold monthly values
                df_monthly = pd.DataFrame({'date': dates, 'value': values})
                
                # Set date column as index
                df_monthly.set_index('date', inplace=True)
                
                # Filter out rows before 2010
                df_monthly = df_monthly[df_monthly.index >= datetime(2010, 1, 1)]
                
                # Resample to daily frequency and interpolate linearly
                df_daily = df_monthly.resample('D').interpolate(method='linear')
                
                # Add the symbol as a new column to identify which series the data belongs to
                df_daily['symbol'] = symbol
                
                # Set the symbol as a column for the final result
                return df_daily[['value']].rename(columns={'value': symbol})  # Use symbol as column name
            else:
                print(f"No valid data found for symbol {symbol}")
                return None
        else:
            print(f"No data available for symbol {symbol}")
            return None
    else:
        print(f"Error: Unable to fetch data for symbol {symbol}, Status Code: {response.status_code}")
        return None

# Initialize an empty list to store the daily data DataFrames
all_data = []

# Loop through all symbols and fetch the data
for symbol in symbols:
    daily_data = fetch_and_interpolate(symbol, api_key)
    if daily_data is not None:
        all_data.append(daily_data)

# Concatenate all the data into one DataFrame, aligning by the index (date)
if all_data:
    final_data = pd.concat(all_data, axis=1)  # Concatenate horizontally by columns

    # Reset the index to add 'date' column for CSV output
    final_data.reset_index(inplace=True)
    
    final_data = final_data.round(3)

    # Save the result to a CSV file, including the 'date' column
    final_data.to_csv('fredData.csv', index=False)
    print("Data successfully written to 'daily_fred_data_after_2010_horizontal.csv'")
else:
    print("No data to write.")
