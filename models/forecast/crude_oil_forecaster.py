import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
import tensorflow as tf

class CrudeOilExogenousForecaster:
    def __init__(self, lookback_period=30):
        """
        Initialize the LSTM forecaster for exogenous variables.
        
        Parameters:
        lookback_period (int): Number of previous time steps to use for prediction
        """
        self.lookback_period = lookback_period
        self.scalers = {}
        self.models = {}
        self.exog_vars = ['Open', 'High', 'Low', 'Volume']
        
    def load_data(self, file_path):
        """Load and prepare the crude oil data"""
        # Read data
        df = pd.read_csv(file_path)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date to ensure chronological order
        df = df.sort_values('Date')
        
        # Set date as index
        df.set_index('Date', inplace=True)
        
        return df
    
    def create_sequences(self, data, target_column):
        """Create sequences for LSTM input"""
        X, y = [], []
        
        for i in range(len(data) - self.lookback_period):
            X.append(data[i:(i + self.lookback_period)])
            y.append(data[i + self.lookback_period, target_column])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train_exogenous_models(self, df, test_size=0.2, epochs=100, batch_size=32, verbose=1):
        """Train separate LSTM models for each exogenous variable"""
        
        # Create a copy of the dataframe with only the exogenous variables and Close
        data = df[self.exog_vars + ['Close']].copy()
        
        # Scale data
        for column in data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[column] = scaler.fit_transform(data[[column]])
            self.scalers[column] = scaler
        
        # Convert to numpy array
        dataset = data.values
        
        # Train models for each exogenous variable
        for i, exog_var in enumerate(self.exog_vars):
            print(f"Training model for {exog_var}...")
            
            # Create sequences
            X, y = self.create_sequences(dataset, i)
            
            # Split into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Build and train model
            model = self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=verbose
            )
            
            # Store model
            self.models[exog_var] = model
            
            # Plot loss
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss for {exog_var}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def forecast_exogenous(self, df, forecast_steps):
        """Forecast exogenous variables for specified number of steps"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_exogenous_models first.")
        
        # Get the most recent data for the lookback period
        recent_data = df[self.exog_vars + ['Close']].iloc[-self.lookback_period:].copy()
        
        # Scale the data
        for column in recent_data.columns:
            recent_data[column] = self.scalers[column].transform(recent_data[[column]])
        
        # Convert to numpy array
        recent_array = recent_data.values
        
        # Initialize forecasts dictionary
        forecasts = {var: [] for var in self.exog_vars}
        
        # Make forecasts for each step
        for step in range(forecast_steps):
            # Create input sequence for prediction
            current_sequence = recent_array[-self.lookback_period:].reshape(1, self.lookback_period, len(recent_data.columns))
            
            # Predict each exogenous variable
            for i, exog_var in enumerate(self.exog_vars):
                # Make prediction
                pred = self.models[exog_var].predict(current_sequence, verbose=0)
                
                # Store scaled prediction
                forecasts[exog_var].append(pred[0, 0])
                
                # Update recent array with prediction for next step
                if step < forecast_steps - 1:  # Don't need to update on the last step
                    new_row = recent_array[-1].copy()
                    new_row[i] = pred[0, 0]
                    recent_array = np.vstack([recent_array, new_row])
        
        # Inverse transform predictions
        for var in self.exog_vars:
            forecasts[var] = self.scalers[var].inverse_transform(
                np.array(forecasts[var]).reshape(-1, 1)
            ).flatten()
        
        # Create forecast dataframe
        dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame(forecasts, index=dates)
        
        return forecast_df
    
    def plot_forecasts(self, df, forecast_df, variable):
        """Plot historical data and forecasts for a specific variable"""
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(df.index[-90:], df[variable][-90:], label='Historical')
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df[variable], 'r--', label='Forecast')
        
        plt.title(f'{variable} - Historical vs Forecast')
        plt.xlabel('Date')
        plt.ylabel(variable)
        plt.legend()
        plt.grid(True)
        plt.show()


def main():
    # Set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize forecaster
    forecaster = CrudeOilExogenousForecaster(lookback_period=30)
    
    # Load data
    file_path = 'data/yahoo_finance_2024.csv'  # Change this to your file path
    df = forecaster.load_data(file_path)
    
    # Train models
    forecaster.train_exogenous_models(df, epochs=50)
    
    # Generate forecasts for next 10 days
    forecast_steps = 30
    forecasts = forecaster.forecast_exogenous(df, forecast_steps)
    
    print("\nForecasted Exogenous Variables:")
    print(forecasts)
    
    # Plot forecasts
    for var in forecaster.exog_vars:
        forecaster.plot_forecasts(df, forecasts, var)
    
    # Save forecasts to CSV
    forecasts.to_csv('exogenous_forecasts.csv')
    print("\nForecasts saved to 'exogenous_forecasts.csv'")


if __name__ == "__main__":
    main()