import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ForecastingDataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        
    def load_raw_data(self):
        print(f"Loading raw OPSD data from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, parse_dates=['utc_timestamp'])
        df.set_index('utc_timestamp', inplace=True)
        df.index = df.index.tz_localize(None)
        print(f"Loaded raw data. Shape: {df.shape}")
        return df

    def extract_country_data(self, df, country):
        print(f"Processing data for {country}...")
        
        configs = {
            'DE': {
                'price': lambda d: d['DE_LU_price_day_ahead'].fillna(d['AT_price_day_ahead']),
                'load': 'DE_load_actual_entsoe_transparency',
                'load_forecast': 'DE_load_forecast_entsoe_transparency',
                'solar': 'DE_solar_generation_actual',
                'wind': lambda d: d['DE_wind_onshore_generation_actual'].fillna(0) + d['DE_wind_offshore_generation_actual'].fillna(0)
            },
            'DK_1': {
                'price': 'DK_1_price_day_ahead',
                'load': 'DK_1_load_actual_entsoe_transparency',
                'load_forecast': 'DK_1_load_forecast_entsoe_transparency',
                'solar': 'DK_1_solar_generation_actual',
                'wind': 'DK_1_wind_generation_actual'
            },
            'GB_GBN': {
                'price': 'GB_GBN_price_day_ahead',
                'load': 'GB_GBN_load_actual_entsoe_transparency',
                'load_forecast': 'GB_GBN_load_forecast_entsoe_transparency',
                'solar': 'GB_GBN_solar_generation_actual',
                'wind': 'GB_GBN_wind_generation_actual'
            }
        }
        
        config = configs[country]
        temp = pd.DataFrame(index=df.index)
        
        # Extract variables
        if callable(config['price']):
            temp['price'] = config['price'](df)
        else:
            temp['price'] = df[config['price']]
            
        temp['load'] = df[config['load']]
        temp['load_forecast'] = df[config['load_forecast']]
        temp['solar'] = df[config['solar']].fillna(0)
        
        if callable(config['wind']):
            temp['wind'] = config['wind'](df)
        else:
            temp['wind'] = df[config['wind']].fillna(0)
            
        # Calculate Net Load and Load Error
        temp['net_load'] = temp['load'] - temp['solar'] - temp['wind']
        temp['load_error'] = (temp['load'] - temp['load_forecast']) / 1000.0  # in GW
        
        # Handle missing prices/loads (interpolate minor gaps)
        temp['price'] = temp['price'].interpolate(method='linear', limit=3)
        temp['load'] = temp['load'].interpolate(method='linear', limit=3)
        temp['load_forecast'] = temp['load_forecast'].interpolate(method='linear', limit=3)
        
        temp.dropna(subset=['price', 'load', 'load_forecast'], inplace=True)
        print(f"  Extracted {country}. Shape: {temp.shape}")
        return temp

    def build_forecasting_dataset(self, df, target_var='price'):
        """
        Builds a daily forecasting dataset where the forecast is made at day d-1 at 12:00 (noon)
        to predict the 24 hours of day d.
        """
        print(f"Building features for target variable: {target_var}...")
        
        # Resample to daily index to iterate over days
        # We need the index to be sorted
        df = df.sort_index()
        
        # Create helper time columns
        df['date'] = df.index.date
        df['hour'] = df.index.hour
        
        # List of unique dates
        all_dates = sorted(df['date'].unique())
        
        # We will build features for each date `target_date` (day d)
        # using information up to `target_date - 1` at 12:00 (noon).
        # Target: 24 values of target_var for `target_date`
        
        rows = []
        timestamps = []
        
        for d_idx in range(2, len(all_dates)):
            target_date = all_dates[d_idx]
            prev_date = all_dates[d_idx - 1]
            prev_prev_date = all_dates[d_idx - 2]
            
            # Target day data
            target_day_data = df[df['date'] == target_date]
            if len(target_day_data) != 24:
                continue # Skip days with incomplete hours (e.g. daylight saving transitions or boundaries)
                
            # Previous day data
            prev_day_data = df[df['date'] == prev_date]
            if len(prev_day_data) != 24:
                continue
                
            # Previous-previous day data
            prev_prev_day_data = df[df['date'] == prev_prev_date]
            if len(prev_prev_day_data) != 24:
                continue
                
            # Feature: target_var values available at noon of prev_date (hours 0 to 12)
            # plus hours of prev_prev_date (hours 0 to 23)
            # We want to represent the available history as of noon on prev_date
            
            # 1. Autoregressive lags from the target variable
            y_prev_noon = prev_day_data.loc[prev_day_data['hour'] == 12, target_var].values[0]
            y_prev_11 = prev_day_data.loc[prev_day_data['hour'] == 11, target_var].values[0]
            y_prev_10 = prev_day_data.loc[prev_day_data['hour'] == 10, target_var].values[0]
            y_prev_09 = prev_day_data.loc[prev_day_data['hour'] == 9, target_var].values[0]
            y_prev_00 = prev_day_data.loc[prev_day_data['hour'] == 0, target_var].values[0]
            
            # Median of the target variable in the last 24h available (noon of day d-2 to noon of day d-1)
            last_24h_slice = pd.concat([
                prev_prev_day_data.loc[prev_prev_day_data['hour'] >= 12],
                prev_day_data.loc[prev_day_data['hour'] <= 12]
            ])[target_var]
            
            y_last_24h_mean = last_24h_slice.mean()
            y_last_24h_std = last_24h_slice.std()
            y_last_24h_median = last_24h_slice.median()
            
            # 2. Exogenous TSO forecasts for the target day (day d)
            # Load forecast for each hour of day d is available on day d-1.
            load_forecasts_d = target_day_data['load_forecast'].values # length 24
            
            # 3. Renewable lags (solar and wind) from the previous days
            solar_prev_noon = prev_day_data.loc[prev_day_data['hour'] == 12, 'solar'].values[0]
            wind_prev_noon = prev_day_data.loc[prev_day_data['hour'] == 12, 'wind'].values[0]
            
            solar_prev_prev_day = prev_prev_day_data['solar'].values # length 24
            wind_prev_prev_day = prev_prev_day_data['wind'].values # length 24
            
            # 4. Same-hour lags from d-2 (prev_prev_day)
            y_d2_hours = prev_prev_day_data[target_var].values # length 24
            
            # 5. Calendar features of day d
            first_hour_dt = target_day_data.index[0]
            dayofweek = first_hour_dt.dayofweek
            month = first_hour_dt.month
            is_weekend = int(dayofweek in [5, 6])
            
            # Fourier features for yearly seasonality (day of year)
            day_of_year = first_hour_dt.dayofyear
            sin_year = np.sin(2 * np.pi * day_of_year / 365.25)
            cos_year = np.cos(2 * np.pi * day_of_year / 365.25)
            
            # Build the feature dictionary
            # Standard features (constant across the target day or matching dimensions)
            base_feat = {
                'dayofweek': dayofweek,
                'month': month,
                'is_weekend': is_weekend,
                'sin_year': sin_year,
                'cos_year': cos_year,
                'y_prev_noon': y_prev_noon,
                'y_prev_11': y_prev_11,
                'y_prev_10': y_prev_10,
                'y_prev_09': y_prev_09,
                'y_prev_00': y_prev_00,
                'y_last_24h_mean': y_last_24h_mean,
                'y_last_24h_std': y_last_24h_std,
                'y_last_24h_median': y_last_24h_median,
                'solar_prev_noon': solar_prev_noon,
                'wind_prev_noon': wind_prev_noon
            }
            
            # We will train 24 separate models.
            # For each hour model h, the row will have:
            # - target variable at hour h of day d
            # - load forecast at hour h of day d
            # - target lag at hour h of day d-2 (same hour lag)
            # - solar/wind lags at hour h of day d-2 (same hour lag)
            # - base features
            for h in range(24):
                feat = base_feat.copy()
                feat['hour'] = h
                feat['load_forecast'] = load_forecasts_d[h]
                feat['y_lag_48'] = y_d2_hours[h]
                feat['solar_lag_48'] = solar_prev_prev_day[h]
                feat['wind_lag_48'] = wind_prev_prev_day[h]
                
                # Target
                feat['target'] = target_day_data.loc[target_day_data['hour'] == h, target_var].values[0]
                feat['target_date'] = str(target_date)
                
                rows.append(feat)
                timestamps.append(target_day_data.index[h])
                
        df_feat = pd.DataFrame(rows, index=timestamps)
        print(f"  Forecasting dataset built. Rows: {len(df_feat)}")
        return df_feat

def main():
    csv_path = os.path.join("data", "time_series_60min_singleindex.csv")
    if os.path.exists(csv_path):
        processor = ForecastingDataProcessor(csv_path)
        raw_df = processor.load_raw_data()
        
        # Test extraction and dataset build for Germany
        de_raw = processor.extract_country_data(raw_df, 'DE')
        de_price_feat = processor.build_forecasting_dataset(de_raw, 'price')
        de_net_load_feat = processor.build_forecasting_dataset(de_raw, 'net_load')
        
        print("\nGermany Price Dataset columns:")
        print(de_price_feat.columns.tolist())
        print(de_price_feat.head(2))
    else:
        print("Data file not found.")

if __name__ == "__main__":
    main()
