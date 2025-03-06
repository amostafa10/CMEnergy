import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings("ignore")

class CrudeOilForecaster:
    def __init__(self, data_path, target_col='Close', exog_cols=None, date_col='Date'):
        """
        Initialize the forecaster with data and configuration.
        
        Parameters:
        - data_path: Path to the CSV file
        - target_col: Column to forecast (default: 'Close')
        - exog_cols: List of exogenous variable columns (default: None)
        - date_col: Name of the date column (default: 'Date')
        """
        self.data_path = data_path
        self.target_col = target_col
        self.exog_cols = exog_cols if exog_cols is not None else []
        self.date_col = date_col
        self.model = None
        self.results = None
        self.df = None
        self.train_data = None
        self.test_data = None
        self.best_params = None
        
    def load_data(self, test_size=0.2):
        """
        Load and prepare data, splitting into train and test sets.
        
        Parameters:
        - test_size: Proportion of data to use for testing (default: 0.2)
        """
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Convert date column to datetime and set as index
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df.set_index(self.date_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Store the dataframe
        self.df = df
        
        # Split into train and test sets
        split_idx = int(len(df) * (1 - test_size))
        self.train_data = df.iloc[:split_idx].copy()
        self.test_data = df.iloc[split_idx:].copy()
        
        print(f"Data loaded: {len(df)} rows")
        print(f"Training data: {len(self.train_data)} rows")
        print(f"Test data: {len(self.test_data)} rows")
        print(f"Training period: {self.train_data.index.min()} to {self.train_data.index.max()}")
        print(f"Testing period: {self.test_data.index.min()} to {self.test_data.index.max()}")
        
        return self.train_data, self.test_data
    
    def add_exogenous_variables(self, new_exog_df, join_type='inner'):
        """
        Add external exogenous variables from another dataframe.
        
        Parameters:
        - new_exog_df: DataFrame containing new exogenous variables
        - join_type: How to join the dataframes ('inner', 'left', etc.)
        
        Returns:
        - Updated DataFrame with new exogenous variables
        """
        if not isinstance(new_exog_df, pd.DataFrame):
            raise ValueError("new_exog_df must be a pandas DataFrame")
        
        # Ensure new_exog_df has a datetime index
        if not isinstance(new_exog_df.index, pd.DatetimeIndex):
            raise ValueError("new_exog_df must have a DatetimeIndex")
        
        # Join the dataframes
        self.df = self.df.join(new_exog_df, how=join_type)
        
        # Update exogenous columns list
        new_cols = list(new_exog_df.columns)
        self.exog_cols.extend(new_cols)
        
        # Re-split into train and test
        if self.train_data is not None and self.test_data is not None:
            split_idx = len(self.train_data)
            self.train_data = self.df.iloc[:split_idx].copy()
            self.test_data = self.df.iloc[split_idx:].copy()
        
        print(f"Added exogenous variables: {new_cols}")
        print(f"Total exogenous variables: {len(self.exog_cols)}")
        
        return self.df
    
    def plot_data(self):
        """
        Plot the target variable and exogenous variables.
        """
        n_exog = len(self.exog_cols)
        fig, axes = plt.subplots(n_exog + 1, 1, figsize=(14, 3 * (n_exog + 1)), sharex=True)
        
        # Handle case with no exogenous variables
        if n_exog == 0:
            axes = [axes]
        
        # Plot target variable
        axes[0].plot(self.df.index, self.df[self.target_col])
        axes[0].set_title(f'Target Variable: {self.target_col}')
        axes[0].grid(True)
        
        # Plot exogenous variables
        for i, col in enumerate(self.exog_cols, 1):
            if col in self.df.columns:
                axes[i].plot(self.df.index, self.df[col])
                axes[i].set_title(f'Exogenous Variable: {col}')
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.savefig('data_visualization.png')
        return fig
    
    def check_stationarity(self, column=None):
        """
        Check stationarity of the target or specified column using ADF test.
        
        Parameters:
        - column: Column to check (default: target column)
        
        Returns:
        - ADF test results
        """
        from statsmodels.tsa.stattools import adfuller
        
        col = column if column is not None else self.target_col
        result = adfuller(self.df[col].dropna())
        
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value}')
        
        if result[1] <= 0.05:
            print(f"Series {col} is stationary")
        else:
            print(f"Series {col} is NOT stationary")
        
        return result
    
    def grid_search(self, p_range=(0, 2), d_range=(0, 2), q_range=(0, 2), 
                    P_range=(0, 2), D_range=(0, 1), Q_range=(0, 2), s=20):
        """
        Perform grid search to find optimal SARIMA parameters.
        
        Parameters:
        - p_range, d_range, q_range: Ranges for ARIMA parameters
        - P_range, D_range, Q_range: Ranges for seasonal ARIMA parameters
        - s: Seasonal period
        
        Returns:
        - Dictionary of best parameters
        """
        print("Starting grid search for optimal parameters...")
        print("This may take some time...")
        
        best_aic = float('inf')
        best_params = None
        
        # Create parameter combinations
        p_params = range(p_range[0], p_range[1] + 1)
        d_params = range(d_range[0], d_range[1] + 1)
        q_params = range(q_range[0], q_range[1] + 1)
        P_params = range(P_range[0], P_range[1] + 1)
        D_params = range(D_range[0], D_range[1] + 1)
        Q_params = range(Q_range[0], Q_range[1] + 1)
        
        # Get exogenous variables if specified
        exog_train = self.train_data[self.exog_cols] if self.exog_cols else None
        
        # Grid search
        for params in itertools.product(p_params, d_params, q_params, P_params, D_params, Q_params):
            p, d, q, P, D, Q = params
            
            # Skip non-invertible models
            if p + d + q + P + D + Q == 0:
                continue
                
            try:
                model = SARIMAX(
                    self.train_data[self.target_col],
                    exog=exog_train,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit(disp=False)
                aic = results.aic
                
                if aic < best_aic:
                    best_aic = aic
                    best_params = {
                        'order': (p, d, q),
                        'seasonal_order': (P, D, Q, s),
                        'aic': aic
                    }
                    print(f"New best parameters: SARIMA{(p,d,q)}x{(P,D,Q,s)} - AIC: {aic}")
            
            except Exception as e:
                continue
        
        self.best_params = best_params
        print(f"Best parameters: SARIMA{best_params['order']}x{best_params['seasonal_order']}")
        print(f"Best AIC: {best_params['aic']}")
        
        return best_params
    
    def train_model(self, order=None, seasonal_order=None, s=20):
        """
        Train the SARIMAX model using the specified or found parameters.
        
        Parameters:
        - order: ARIMA order (p, d, q)
        - seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        - s: Seasonal period
        
        Returns:
        - Trained model results
        """
        # If parameters are not provided, use best parameters from grid search
        if order is None and self.best_params is not None:
            order = self.best_params['order']
        elif order is None:
            order = (1, 1, 1)  # Default
            
        if seasonal_order is None and self.best_params is not None:
            seasonal_order = self.best_params['seasonal_order']
        elif seasonal_order is None:
            seasonal_order = (1, 1, 1, s)  # Default
            
        # Get exogenous variables if specified
        exog_train = self.train_data[self.exog_cols] if self.exog_cols and self.exog_cols[0] in self.train_data.columns else None
            
        print(f"Training SARIMA{order}x{seasonal_order} model...")
        
        self.model = SARIMAX(
            self.train_data[self.target_col],
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.results = self.model.fit(disp=False)
        print("Model training complete.")
        print(self.results.summary())
        
        return self.results
    
    def evaluate_model(self):
        """
        Evaluate the model on test data.
        
        Returns:
        - Dictionary of evaluation metrics
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet.")
            
        # Get exogenous variables for test set if specified
        exog_test = self.test_data[self.exog_cols] if self.exog_cols and self.exog_cols[0] in self.test_data.columns else None
        
        # Use the start parameter based on the length of the training data
        # This resolves the KeyError by using the correct prediction start point
        start = len(self.train_data)
        end = start + len(self.test_data) - 1
            
        # Make predictions
        predictions = self.results.predict(
            start=start,
            end=end,
            exog=exog_test
        )
        
        # Assign predictions to the test date index for better comparison
        predictions.index = self.test_data.index
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(self.test_data[self.target_col], predictions))
        mae = mean_absolute_error(self.test_data[self.target_col], predictions)
        mape = np.mean(np.abs((self.test_data[self.target_col] - predictions) / self.test_data[self.target_col])) * 100
        
        # Store and print metrics
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        print("Model Evaluation Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.4f}%")
        
        # Plot actual vs predicted
        plt.figure(figsize=(12, 6))
        plt.plot(self.test_data.index, self.test_data[self.target_col], label='Actual')
        plt.plot(self.test_data.index, predictions, label='Predicted', color='red')
        plt.title('SARIMAX Forecast vs Actual')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.savefig('forecast_evaluation.png')
        
        return metrics, predictions
    
    def forecast(self, steps=30, exog_future=None, plot=True):
        """
        Generate forecasts for future periods.
        
        Parameters:
        - steps: Number of steps to forecast
        - exog_future: DataFrame of future exogenous variables
        - plot: Whether to plot the forecast
        
        Returns:
        - DataFrame with forecasts
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet.")
            
        # Check if we have exogenous variables in the model
        if self.exog_cols and len(self.exog_cols) > 0 and exog_future is None:
            print("Warning: Model was trained with exogenous variables, but no future values provided.")
            print("Generating future exogenous values using ARIMA method...")
            exog_future = self.generate_future_exog(steps=steps, method='arima')
            
        # Generate forecast
        print(f"Generating forecast for {steps} periods...")
        
        # Get the last observed values for plotting
        history = self.df[self.target_col]
        
        # Get start point for forecasting (right after the end of our data)
        forecast_start = len(self.df)
        
        # Forecast
        forecast = self.results.get_forecast(steps=steps, exog=exog_future.dropna())
        
        # Create forecast index
        last_date = self.df.index[-1]
        freq = pd.infer_freq(self.df.index)
        if freq is None:
            # Try to determine frequency from consecutive dates
            if len(self.df) > 1:
                # Calculate the most common difference between dates
                date_diffs = self.df.index[1:] - self.df.index[:-1]
                most_common_diff = pd.Series(date_diffs).value_counts().index[0]
                freq = most_common_diff
            else:
                # Default to business days if we can't determine
                freq = 'B'
                
        forecast_idx = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )
        
        # Get forecast mean and confidence intervals
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'forecast': mean_forecast,
            'lower_ci': conf_int.iloc[:, 0],
            'upper_ci': conf_int.iloc[:, 1]
        }, index=forecast_idx)
        
        # Plot the forecast if requested
        if plot:
            plt.figure(figsize=(12, 6))
            
            # Plot history (last 60 days for better visualization)
            plt.plot(history.index[-60:], history.iloc[-60:], label='Historical')
            
            # Plot forecast
            plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='red')
            
            # Plot confidence intervals
            plt.fill_between(
                forecast_df.index,
                forecast_df['lower_ci'],
                forecast_df['upper_ci'],
                color='pink', alpha=0.3
            )
            
            plt.title(f'{steps}-Period Forecast for {self.target_col}')
            plt.xlabel('Date')
            plt.ylabel(self.target_col)
            plt.legend()
            plt.grid(True)
            plt.savefig('future_forecast.png')
        
        print("Forecast generated successfully.")
        return forecast_df
    
    def generate_future_exog(self, steps=30, method='last'):
        """
        Generate future values for exogenous variables.
        
        Parameters:
        - steps: Number of steps to generate
        - method: Method to use ('last', 'mean', 'arima')
        
        Returns:
        - DataFrame of future exogenous values
        """
        if not self.exog_cols or len(self.exog_cols) == 0:
            return None
            
        print(f"Generating future exogenous values using '{method}' method...")
        
        # Create future date index
        last_date = self.df.index[-1]
        freq = pd.infer_freq(self.df.index)
        if freq is None:
            # Try to determine frequency from consecutive dates
            if len(self.df) > 1:
                # Calculate the most common difference between dates
                date_diffs = self.df.index[1:] - self.df.index[:-1]
                most_common_diff = pd.Series(date_diffs).value_counts().index[0]
                freq = most_common_diff
            else:
                # Default to business days if we can't determine
                freq = 'B'
                
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq=freq
        )
        
        exog_future = pd.DataFrame(index=future_dates)
        
        for col in self.exog_cols:
            if col not in self.df.columns:
                continue
                
            if method == 'last':
                # Use the last value for all future periods
                exog_future[col] = self.df[col].iloc[-1]
                
            elif method == 'mean':
                # Use the mean of the last 30 values
                window = min(30, len(self.df))
                exog_future[col] = self.df[col].iloc[-window:].mean()
                
            elif method == 'arima':
                # Use ARIMA to forecast exogenous variables
                from statsmodels.tsa.arima.model import ARIMA
                
                model = ARIMA(self.df[col], order=(1, 1, 1))
                model_fit = model.fit()
                exog_forecast = model_fit.forecast(steps=steps)
                exog_future[col] = exog_forecast
        
        return exog_future
    
    def analyze_diagnostics(self):
        """
        Plot model diagnostic plots to check model assumptions.
        
        Returns:
        - Figure with diagnostic plots
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet.")
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Standardized residuals
        self.results.plot_diagnostics(fig=fig)
        plt.tight_layout()
        plt.savefig('model_diagnostics.png')
        
        return fig
    
    def plot_components(self):
        """
        Plot the SARIMAX components if available.
        
        Returns:
        - Figure with component plots
        """
        if self.results is None:
            raise ValueError("Model has not been trained yet.")
        
        try:
            # Check if the model has component attributes
            components = {}
            
            if hasattr(self.results, 'level'):
                components['level'] = self.results.level
                
            if hasattr(self.results, 'trend'):
                components['trend'] = self.results.trend
                
            if hasattr(self.results, 'season'):
                components['seasonal'] = self.results.season
                
            if hasattr(self.results, 'resid'):
                components['irregular'] = self.results.resid
                
            # If no components are available, extract them from the results
            if not components:
                # Get in-sample predictions to use as trend
                fittedvalues = self.results.fittedvalues
                components['trend'] = fittedvalues
                
                # Calculate residuals
                residuals = self.train_data[self.target_col] - fittedvalues
                components['irregular'] = residuals
                
                # Create a seasonal component using the decomposition method
                # (this is an approximation)
                try:
                    seasonal_period = self.results.specification['seasonal_order'][3]
                    
                    if hasattr(seasonal_decompose, 'seasonal_decompose'):
                        decomp = seasonal_decompose(self.train_data[self.target_col], period=seasonal_period)
                        components['seasonal'] = decomp.seasonal
                except:
                    components['seasonal'] = pd.Series(0, index=self.train_data.index)  # No seasonal component
                
                # Set level as original values
                components['level'] = self.train_data[self.target_col]
                
            # Convert to DataFrame
            df_components = pd.DataFrame(components, index=self.train_data.index)
            
            # Plot components
            n_components = len(df_components.columns)
            fig, axes = plt.subplots(n_components, 1, figsize=(14, 3*n_components))
            
            # Handle case with a single component
            if n_components == 1:
                axes = [axes]
                
            for i, col in enumerate(df_components.columns):
                df_components[col].plot(ax=axes[i])
                axes[i].set_title(f'{col.capitalize()} Component')
                axes[i].grid(True)
            
            plt.tight_layout()
            plt.savefig('model_components.png')
            
            return fig, df_components
            
        except Exception as e:
            print(f"Warning: Could not extract all time series components: {str(e)}")
            # If components aren't available, just show the fitted values vs actuals
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.train_data.index, self.train_data[self.target_col], label='Actual')
            ax.plot(self.train_data.index, self.results.fittedvalues, label='Fitted')
            ax.set_title(f'Actual vs Fitted Values for {self.target_col}')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig('fitted_vs_actual.png')
            
            return fig, None

# Example usage
def main():
    # Initialize forecaster
    forecaster = CrudeOilForecaster(
        data_path='data\\yahoo_finance_2024.csv',
        target_col='Close',
        exog_cols=['Volume', 'Open', 'High', 'Low']
    )
    
    # Load data
    train_data, test_data = forecaster.load_data(test_size=0.2)
    
    # Plot the data
    forecaster.plot_data()
    
    # Check stationarity
    forecaster.check_stationarity()
    
    # Example: Add macroeconomic data if available
    # macro_data = pd.read_csv('macro_data.csv', parse_dates=['Date'], index_col='Date')
    # forecaster.add_exogenous_variables(macro_data)
    
    # Option 1: Use grid search to find optimal parameters (time-consuming)
    # best_params = forecaster.grid_search(
    #     p_range=(0, 2), d_range=(0, 2), q_range=(0, 2),
    #     P_range=(0, 1), D_range=(0, 1), Q_range=(0, 1),
    #     s=20  # Assuming monthly seasonality (20 trading days)
    # )
    
    # Option 2: Specify parameters directly
    # Train the model with specified parameters
    forecaster.train_model(order=(1, 1, 1), seasonal_order=(1, 1, 1, 20))
    
    # Analyze model diagnostics
    forecaster.analyze_diagnostics()
    
    # Plot components
    # forecaster.plot_components()
    
    # Evaluate the model
    metrics, predictions = forecaster.evaluate_model()
    
    # Generate future exogenous values
    future_exog = forecaster.generate_future_exog(steps=30, method='arima')
    
    # Generate forecast
    forecast = forecaster.forecast(steps=30, exog_future=future_exog)
    
    print("\nSARIMAX forecasting completed successfully!")
    print(f"\nForecast for the next 30 periods:\n{forecast.head()}")

if __name__ == "__main__":
    main()