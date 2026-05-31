import os
import pandas as pd
import numpy as np

class OPSDDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Loads the OPSD CSV file, parses the timestamp index.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found at {self.file_path}")
        
        print(f"Loading OPSD data from {self.file_path}...")
        # Load and parse timestamps
        self.df = pd.read_csv(self.file_path)
        self.df['utc_timestamp'] = pd.to_datetime(self.df['utc_timestamp'])
        self.df.set_index('utc_timestamp', inplace=True)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def process_germany_data(self):
        """
        Processes and extracts German-specific variables:
        - Combines DE_LU and AT prices to form a continuous price series.
        - Forward-fills solar and wind capacities.
        - Extracts load, solar, wind, price, and capacities.
        """
        if self.df is None:
            self.load_data()

        # Combine prices
        # Before Oct 2018, AT and DE shared the bidding zone, price is in AT_price_day_ahead.
        # After Oct 2018, it is in DE_LU_price_day_ahead.
        price_de_lu = self.df['DE_LU_price_day_ahead'] if 'DE_LU_price_day_ahead' in self.df.columns else pd.Series(np.nan, index=self.df.index)
        price_at = self.df['AT_price_day_ahead'] if 'AT_price_day_ahead' in self.df.columns else pd.Series(np.nan, index=self.df.index)
        combined_price = price_de_lu.fillna(price_at)

        # Extract other core variables
        solar_gen = self.df['DE_solar_generation_actual']
        solar_cap = self.df['DE_solar_capacity']
        
        # Wind: combine onshore and offshore actuals
        wind_onshore = self.df['DE_wind_onshore_generation_actual'] if 'DE_wind_onshore_generation_actual' in self.df.columns else pd.Series(0, index=self.df.index)
        wind_offshore = self.df['DE_wind_offshore_generation_actual'] if 'DE_wind_offshore_generation_actual' in self.df.columns else pd.Series(0, index=self.df.index)
        wind_gen = wind_onshore.fillna(0) + wind_offshore.fillna(0)
        
        # Wind capacity: combine onshore and offshore capacities
        wind_onshore_cap = self.df['DE_wind_onshore_capacity'] if 'DE_wind_onshore_capacity' in self.df.columns else pd.Series(0, index=self.df.index)
        wind_offshore_cap = self.df['DE_wind_offshore_capacity'] if 'DE_wind_offshore_capacity' in self.df.columns else pd.Series(0, index=self.df.index)
        wind_cap = wind_onshore_cap.fillna(0) + wind_offshore_cap.fillna(0)

        # Load actual and forecast
        load_actual = self.df['DE_load_actual_entsoe_transparency']
        load_forecast = self.df['DE_load_forecast_entsoe_transparency']

        # Construct dataframe
        data = pd.DataFrame({
            'price': combined_price,
            'load': load_actual,
            'load_forecast': load_forecast,
            'solar': solar_gen,
            'solar_capacity': solar_cap,
            'wind': wind_gen,
            'wind_capacity': wind_cap
        }, index=self.df.index)

        # Forward fill capacity values (capacities are slow-moving step functions, filling is physically correct)
        data['solar_capacity'] = data['solar_capacity'].ffill().bfill()
        data['wind_capacity'] = data['wind_capacity'].ffill().bfill()

        # Net load
        data['net_load'] = data['load'] - data['solar'] - data['wind']
        data['net_load_forecast'] = data['load_forecast'] - data['solar'] - data['wind']

        print(f"German variables processed. Shape: {data.shape}")
        return data

    def build_features(self, df):
        """
        Creates temporal and lag features for day-ahead price forecasting.
        """
        print("Building temporal and lag features...")
        df_feat = df.copy()

        # Calendar features
        df_feat['hour'] = df_feat.index.hour
        df_feat['dayofweek'] = df_feat.index.dayofweek
        df_feat['month'] = df_feat.index.month
        df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
        
        # Lag features (lagged target and inputs)
        # In day-ahead forecasting, tomorrow's price is predicted.
        # Today's price (lag 24) and last week's price (lag 168) are used.
        df_feat['price_lag_24'] = df_feat['price'].shift(24)
        df_feat['price_lag_48'] = df_feat['price'].shift(48)
        df_feat['price_lag_168'] = df_feat['price'].shift(168)

        # Input lags
        df_feat['load_lag_24'] = df_feat['load'].shift(24)
        df_feat['solar_lag_24'] = df_feat['solar'].shift(24)
        df_feat['wind_lag_24'] = df_feat['wind'].shift(24)

        # Feature Interactions for Capacity-Conditioned Price Cannibalization
        # Generation divided by capacity represents the capacity factor (a value between 0 and 1)
        df_feat['solar_cap_factor'] = df_feat['solar'] / df_feat['solar_capacity']
        df_feat['wind_cap_factor'] = df_feat['wind'] / df_feat['wind_capacity']

        # Cross terms to model the interaction of generation and capacity:
        # As solar capacity grows, high solar capacity factors drive price down even more.
        df_feat['solar_interaction'] = df_feat['solar'] * df_feat['solar_capacity'] / 1e6
        df_feat['wind_interaction'] = df_feat['wind'] * df_feat['wind_capacity'] / 1e6

        # Drop rows with NaN values created by lagging
        df_feat.dropna(subset=['price', 'price_lag_24', 'price_lag_168', 'load_forecast'], inplace=True)
        print(f"Features created successfully. Shape after dropping NaNs: {df_feat.shape}")
        return df_feat

if __name__ == "__main__":
    import sys
    csv_path = os.path.join(os.path.dirname(__file__), "data", "time_series_60min_singleindex.csv")
    if os.path.exists(csv_path):
        processor = OPSDDataProcessor(csv_path)
        raw_df = processor.process_germany_data()
        feat_df = processor.build_features(raw_df)
        print(feat_df.head())
    else:
        print("Data file not found. Run main.py or data_downloader.py first.")
