import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.api.layers import Dense
from keras.api.models import Sequential, Model
from keras.api.layers import LSTM, Dense, Dropout, Input, Concatenate
from keras.api.callbacks import EarlyStopping
import tensorflow as tf
from keras.api import backend as K

class EnhancedCrudeOilForecaster:
    def __init__(self, lookback_period=30):
        self.lookback_period = lookback_period
        self.scalers = {}
        self.models = {}
        self.volatility_models = {}
        self.exog_vars = ['Open', 'High', 'Low', 'Volume']
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)
        
        for col in self.exog_vars:
            df[f'{col}_change'] = df[col].pct_change()
            df[f'{col}_volatility'] = df[col].rolling(window=20).std()
            df[f'{col}_SMA5'] = df[col].rolling(window=5).mean()
            df[f'{col}_SMA20'] = df[col].rolling(window=20).mean()
            
        df = df.dropna()
        return df
    
    def create_sequences(self, data, target_column, include_volatility=False):
        X, y = [], []
        
        if include_volatility:
            for i in range(len(data) - self.lookback_period - 5):
                seq = data[i:(i + self.lookback_period)]
                X.append(seq)
                next_5_vals = data[i + self.lookback_period:i + self.lookback_period + 5, target_column]
                volatility = np.std(next_5_vals)
                y.append(volatility)
        else:
            for i in range(len(data) - self.lookback_period):
                X.append(data[i:(i + self.lookback_period)])
                y.append(data[i + self.lookback_period, target_column])
            
        return np.array(X), np.array(y)
    
    def build_advanced_model(self, input_shape):
        main_input = Input(shape=input_shape)
        lstm_1 = LSTM(units=64, return_sequences=True)(main_input)
        dropout_1 = Dropout(0.3)(lstm_1)
        lstm_2 = LSTM(units=64)(dropout_1)
        dropout_2 = Dropout(0.3)(lstm_2)
        main_output = Dense(units=1, name='main_output')(dropout_2)
        model = Model(inputs=main_input, outputs=main_output)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_volatility_model(self, input_shape):
        model = Sequential()
        model.add(LSTM(units=32, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='relu'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def tilted_loss(self, q, y_true, y_pred):
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e))
    
    def add_technical_indicators(self, df):
        features = df.copy()
        
        for col in self.exog_vars:
            rolling_mean = features[col].rolling(window=20).mean()
            rolling_std = features[col].rolling(window=20).std()
            features[f'{col}_bollinger_upper'] = rolling_mean + (rolling_std * 2)
            features[f'{col}_bollinger_lower'] = rolling_mean - (rolling_std * 2)
            
            ema12 = features[col].ewm(span=12).mean()
            ema26 = features[col].ewm(span=26).mean()
            features[f'{col}_macd'] = ema12 - ema26
            features[f'{col}_macd_signal'] = features[f'{col}_macd'].ewm(span=9).mean()
            
            delta = features[col].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = -delta.clip(upper=0).rolling(window=14).mean()
            rs = gain / loss
            features[f'{col}_rsi'] = 100 - (100 / (1 + rs))
        
        features = features.dropna()
        return features
    
    def train_models_with_volatility(self, df, test_size=0.2, epochs=100, batch_size=32, verbose=1):
        enhanced_df = self.add_technical_indicators(df)
        feature_cols = list(enhanced_df.columns)
        data = enhanced_df[feature_cols].copy()
        
        for column in data.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[column] = scaler.fit_transform(data[[column]])
            self.scalers[column] = scaler
        
        dataset = data.values
        
        for i, exog_var in enumerate(self.exog_vars):
            print(f"Training model for {exog_var}...")
            col_idx = feature_cols.index(exog_var)
            X, y = self.create_sequences(dataset, col_idx)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            model = self.build_advanced_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=verbose
            )
            
            self.models[exog_var] = model
            
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title(f'Training and Validation Loss for {exog_var}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            print(f"Training volatility model for {exog_var}...")
            X_vol, y_vol = self.create_sequences(dataset, col_idx, include_volatility=True)
            X_vol_train, X_vol_test, y_vol_train, y_vol_test = train_test_split(
                X_vol, y_vol, test_size=test_size, shuffle=False
            )
            
            vol_model = self.build_volatility_model((X_vol_train.shape[1], X_vol_train.shape[2]))
            
            vol_model.fit(
                X_vol_train, y_vol_train,
                epochs=epochs//2,
                batch_size=batch_size,
                validation_data=(X_vol_test, y_vol_test),
                callbacks=[early_stopping],
                verbose=verbose
            )
            
            self.volatility_models[exog_var] = vol_model

    def forecast_with_monte_carlo(self, df, forecast_steps, num_simulations=100):
        if not self.models:
            raise ValueError("Models not trained. Call train_models_with_volatility first.")
        
        enhanced_df = self.add_technical_indicators(df)
        feature_cols = list(enhanced_df.columns)
        recent_data = enhanced_df.iloc[-self.lookback_period:].copy()
        
        for column in recent_data.columns:
            recent_data[column] = self.scalers[column].transform(recent_data[[column]])
        
        recent_array = recent_data.values
        all_simulations = {var: np.zeros((num_simulations, forecast_steps)) for var in self.exog_vars}
        
        for sim in range(num_simulations):
            sim_array = recent_array.copy()
            
            for step in range(forecast_steps):
                current_sequence = sim_array[-self.lookback_period:].reshape(1, self.lookback_period, sim_array.shape[1])
                step_preds = {}
                
                for var in self.exog_vars:
                    col_idx = feature_cols.index(var)
                    baseline_pred = self.models[var].predict(current_sequence, verbose=0)[0, 0]
                    volatility = self.volatility_models[var].predict(current_sequence, verbose=0)[0, 0]
                    noise = np.random.normal(0, volatility)
                    prediction = baseline_pred + noise
                    step_preds[var] = prediction
                    all_simulations[var][sim, step] = prediction
                
                if step < forecast_steps - 1:
                    new_row = sim_array[-1].copy()
                    
                    for var in self.exog_vars:
                        col_idx = feature_cols.index(var)
                        new_row[col_idx] = step_preds[var]
                    
                    sim_array = np.vstack([sim_array, new_row])
        
        forecasts = {}
        confidence_intervals = {}
        
        for var in self.exog_vars:
            mean_forecast = np.mean(all_simulations[var], axis=0)
            lower_bound = np.percentile(all_simulations[var], 5, axis=0)
            upper_bound = np.percentile(all_simulations[var], 95, axis=0)
            
            mean_forecast = self.scalers[var].inverse_transform(mean_forecast.reshape(-1, 1)).flatten()
            lower_bound = self.scalers[var].inverse_transform(lower_bound.reshape(-1, 1)).flatten()
            upper_bound = self.scalers[var].inverse_transform(upper_bound.reshape(-1, 1)).flatten()
            
            forecasts[var] = mean_forecast
            confidence_intervals[var] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)
        forecast_df = pd.DataFrame(forecasts, index=dates)
        lower_df = pd.DataFrame({f"{var}_lower": confidence_intervals[var]['lower'] for var in self.exog_vars}, index=dates)
        upper_df = pd.DataFrame({f"{var}_upper": confidence_intervals[var]['upper'] for var in self.exog_vars}, index=dates)
        result_df = pd.concat([forecast_df, lower_df, upper_df], axis=1)
        
        return result_df
    
    def plot_forecasts_with_confidence(self, df, forecast_df, days_to_show=90):
        for var in self.exog_vars:
            plt.figure(figsize=(14, 7))
            historical = df[var].iloc[-days_to_show:]
            plt.plot(historical.index, historical, label='Historical', color='blue')
            plt.plot(forecast_df.index, forecast_df[var], 'r--', label='Forecast', linewidth=2)
            plt.fill_between(
                forecast_df.index,
                forecast_df[f"{var}_lower"],
                forecast_df[f"{var}_upper"],
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
            plt.title(f'Exogenous Variable: {var}', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(var, fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Initialize forecaster
    forecaster = EnhancedCrudeOilForecaster(lookback_period=30)
    
    # Load data
    file_path = 'data/yahoo_finance_2024.csv'
    df = forecaster.load_data(file_path)
    
    # Train models
    forecaster.train_models_with_volatility(df, epochs=50)
    
    # Generate forecasts for next 60 days with Monte Carlo simulation
    forecast_steps = 60
    forecasts = forecaster.forecast_with_monte_carlo(df, forecast_steps, num_simulations=200)
    
    print("\nForecasted Exogenous Variables with Confidence Intervals:")
    print(forecasts.head())
    
    # Plot forecasts with confidence intervals
    forecaster.plot_forecasts_with_confidence(df, forecasts)
    
    # Save forecasts to CSV
    forecasts.to_csv('exogenous_forecasts_with_confidence.csv')
    print("\nForecasts saved to 'exogenous_forecasts_with_confidence.csv'")

if __name__ == "__main__":
    main()