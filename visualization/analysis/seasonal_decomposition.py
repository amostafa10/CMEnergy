import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates

# Load the data
def load_data(file_path):
    """
    Load time series data from a CSV file and prepare it for analysis.
    """
    df = pd.read_csv(file_path)
    
    # Convert 'Date' to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set 'Date' as index
    df.set_index('Date', inplace=True)
    
    # Sort index to ensure chronological order
    df.sort_index(inplace=True)
    
    return df

# Plot the original time series
def plot_original_series(df):
    """
    Plot the original time series data.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title('Crude Oil Close Price Time Series')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    return plt

# Decompose the time series
def decompose_time_series(df, column='Close', period=None, model='additive'):
    """
    Decompose the time series into trend, seasonal, and residual components.
    
    Parameters:
    - df: DataFrame with time series data
    - column: Column to decompose
    - period: Seasonal period (if None, it will be estimated)
    - model: 'additive' or 'multiplicative'
    
    Returns:
    - result: Decomposition result object
    """
    # If period is not provided, estimate it based on data frequency
    if period is None:
        # For daily data, can use different periods like 7 (weekly), 30 (monthly)
        # For financial data, 5 or 20 are common (business week or month)
        period = 20  # Assuming 20 trading days per month
    
    # Perform decomposition
    result = seasonal_decompose(df[column], model=model, period=period)
    
    return result

# Plot the decomposition components
def plot_decomposition(result, title="Time Series Decomposition"):
    """
    Plot the decomposition components.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # Original
    axes[0].plot(result.observed)
    axes[0].set_title('Original Time Series')
    axes[0].grid(True)
    
    # Trend
    axes[1].plot(result.trend)
    axes[1].set_title('Trend Component')
    axes[1].grid(True)
    
    # Seasonal
    axes[2].plot(result.seasonal)
    axes[2].set_title('Seasonal Component')
    axes[2].grid(True)
    
    # Residual
    axes[3].plot(result.resid)
    axes[3].set_title('Residual Component (Noise)')
    axes[3].grid(True)
    
    # Format x-axis dates
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)
    return fig

# Optional: Analyze the residuals
def analyze_residuals(result):
    """
    Analyze the residuals to check if they are white noise.
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy import stats
    
    # Create a DataFrame from the residuals
    resid_df = pd.DataFrame(result.resid.dropna())
    
    # Plot histogram and QQ plot of residuals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram
    axes[0, 0].hist(resid_df.values, bins=30)
    axes[0, 0].set_title('Histogram of Residuals')
    
    # QQ plot
    stats.probplot(resid_df.values.flatten(), dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot of Residuals')
    
    # ACF
    plot_acf(resid_df.values, ax=axes[1, 0])
    
    # PACF
    plot_pacf(resid_df.values, ax=axes[1, 1])
    
    plt.tight_layout()
    return fig

# Main function
def main():
    # Set the file path
    file_path = 'data\\yahoo_finance_2024.csv'  # Replace with your actual file path
    
    try:
        # Load the data
        df = load_data(file_path)
        
        # Display basic information
        print("Data Information:")
        print(f"Date Range: {df.index.min()} to {df.index.max()}")
        print(f"Number of observations: {len(df)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Plot original time series
        original_plot = plot_original_series(df)
        original_plot.savefig('crude_oil_original.png')
        
        # Try both additive and multiplicative decomposition
        # For financial data, multiplicative often works better
        for model in ['additive', 'multiplicative']:
            # Decompose with monthly seasonality (approx. 20 trading days)
            result = decompose_time_series(df, period=20, model=model)
            
            # Plot decomposition
            decomp_plot = plot_decomposition(result, title=f"Crude Oil Price {model.capitalize()} Decomposition")
            decomp_plot.savefig(f'crude_oil_{model}_decomposition.png')
            
            # Analyze residuals
            resid_plot = analyze_residuals(result)
            resid_plot.savefig(f'crude_oil_{model}_residuals.png')
        
        print("\nDecomposition analysis completed. Plots saved.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()