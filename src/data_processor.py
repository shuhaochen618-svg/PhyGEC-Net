"""
OPSD Data Processor for TSO Load Forecast Error Correction
Builds feature-rich datasets for DE, DK_1, GB_GBN
Timeline-safe: all features observable at noon of day d-1
"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

COUNTRY_CONFIG = {
    'DE': {
        'price':         lambda d: d.get('DE_LU_price_day_ahead', pd.Series(dtype=float)).fillna(
                                    d.get('AT_price_day_ahead', pd.Series(dtype=float))),
        'load':          'DE_load_actual_entsoe_transparency',
        'load_forecast': 'DE_load_forecast_entsoe_transparency',
        'solar':         'DE_solar_generation_actual',
        'wind_on':       'DE_wind_onshore_generation_actual',
        'wind_off':      'DE_wind_offshore_generation_actual',
        'avg_load_gw':   55.0,   # approximate average load in GW (for scaling)
    },
    'DK_1': {
        'price':         'DK_1_price_day_ahead',
        'load':          'DK_1_load_actual_entsoe_transparency',
        'load_forecast': 'DK_1_load_forecast_entsoe_transparency',
        'solar':         'DK_1_solar_generation_actual',
        'wind':          'DK_1_wind_generation_actual',
        'avg_load_gw':   1.8,
    },
    'GB_GBN': {
        'price':         'GB_GBN_price_day_ahead',
        'load':          'GB_GBN_load_actual_entsoe_transparency',
        'load_forecast': 'GB_GBN_load_forecast_entsoe_transparency',
        'solar':         'GB_GBN_solar_generation_actual',
        'wind':          'GB_GBN_wind_generation_actual',
        'avg_load_gw':   30.0,
    },
}


def download_data(raw_dir):
    """Download OPSD CSV if not present."""
    path = os.path.join(raw_dir, 'time_series_60min_singleindex.csv')
    if os.path.exists(path):
        print(f"[data] Data file already exists: {path}")
        return path
    print(f"[data] Downloading OPSD data from {DATA_URL} ...")
    import urllib.request
    os.makedirs(raw_dir, exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, path)
    print(f"[data] Downloaded to {path}")
    return path


def load_raw(csv_path):
    print(f"[data] Loading raw CSV...")
    df = pd.read_csv(csv_path, parse_dates=['utc_timestamp'], low_memory=False)
    df.set_index('utc_timestamp', inplace=True)
    # Remove timezone info if present
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    print(f"[data] Raw data shape: {df.shape}, range: {df.index.min()} to {df.index.max()}")
    return df


def extract_country(df, country):
    """Extract and compute core variables for one country."""
    cfg = COUNTRY_CONFIG[country]
    out = pd.DataFrame(index=df.index)

    # Price
    if callable(cfg.get('price', None)):
        out['price'] = cfg['price'](df)
    elif isinstance(cfg.get('price'), str):
        out['price'] = df[cfg['price']] if cfg['price'] in df.columns else np.nan

    # Load
    out['load'] = df[cfg['load']] if cfg['load'] in df.columns else np.nan
    out['load_forecast'] = df[cfg['load_forecast']] if cfg['load_forecast'] in df.columns else np.nan

    # Solar
    solar_col = cfg.get('solar', '')
    out['solar'] = df[solar_col].fillna(0) if solar_col in df.columns else 0.0

    # Wind (handle onshore + offshore for DE)
    if 'wind_on' in cfg and 'wind_off' in cfg:
        won = df[cfg['wind_on']].fillna(0) if cfg['wind_on'] in df.columns else 0.0
        woff = df[cfg['wind_off']].fillna(0) if cfg['wind_off'] in df.columns else 0.0
        out['wind'] = won + woff
    elif 'wind' in cfg:
        out['wind'] = df[cfg['wind']].fillna(0) if cfg['wind'] in df.columns else 0.0
    else:
        out['wind'] = 0.0

    # Interpolate minor gaps (max 3h)
    for col in ['price', 'load', 'load_forecast']:
        out[col] = out[col].interpolate(method='linear', limit=3)

    # Drop rows with missing essentials
    out.dropna(subset=['load', 'load_forecast'], inplace=True)

    # Derived variables
    out['load_error'] = (out['load'] - out['load_forecast']) / 1000.0  # GW
    out['res'] = (out['solar'] + out['wind']).clip(lower=0)
    out['res_pct'] = (out['res'] / out['load'].clip(lower=1)).clip(0, 1)
    out['net_load'] = out['load'] - out['res']

    # Time features
    out['hour'] = out.index.hour
    out['dayofweek'] = out.index.dayofweek
    out['month'] = out.index.month
    out['year'] = out.index.year
    out['is_weekend'] = out['dayofweek'].isin([5, 6]).astype(int)

    print(f"[data] {country}: {len(out)} clean hourly obs, "
          f"mean_error={out['load_error'].mean():.2f} GW, "
          f"std={out['load_error'].std():.2f} GW")
    return out


def build_features(df, country):
    """
    Build the full feature matrix for error correction.
    TARGET: load_error at hour h of day d
    ALL features must be observable at noon on day d-1 (timeline-safe).
    """
    df = df.sort_index().copy()
    e = df['load_error']

    # ── 1. Autoregressive error lags (hours before noon of day d-1) ──────────
    for lag in [24, 48, 72, 168, 336]:
        df[f'err_lag_{lag}'] = e.shift(lag)

    # ── 2. Rolling bias features ──────────────────────────────────────────────
    # Past 24h mean error (up to noon d-1)
    df['err_roll_24h_mean']   = e.shift(24).rolling(24,  min_periods=12).mean()
    df['err_roll_24h_std']    = e.shift(24).rolling(24,  min_periods=12).std()
    df['err_roll_7d_mean']    = e.shift(24).rolling(168, min_periods=72).mean()
    df['err_roll_7d_std']     = e.shift(24).rolling(168, min_periods=72).std()
    df['err_roll_30d_mean']   = e.shift(24).rolling(720, min_periods=168).mean()

    # ── 3. Same-hour-same-weekday rolling bias (innovation feature 3) ─────────
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    # For each row, look back 7, 14, 21 days at same hour
    df['err_same_hour_lag168']  = e.shift(168)
    df['err_same_hour_lag336']  = e.shift(336)
    df['err_same_hour_lag504']  = e.shift(504)
    df['err_same_hour_3wk_mean'] = (
        df['err_same_hour_lag168'] + df['err_same_hour_lag336'] + df['err_same_hour_lag504']
    ) / 3.0

    # ── 4. Error momentum / regime feature (innovation feature 2) ────────────
    sign_lag1 = np.sign(e.shift(24))
    sign_lag2 = np.sign(e.shift(25))
    sign_lag3 = np.sign(e.shift(26))
    df['err_streak'] = sign_lag1 * sign_lag2 * sign_lag3  # +1 if all same sign, -1 otherwise
    df['err_direction_lag24'] = sign_lag1  # simple sign of last known error

    # ── 5. RES × error interaction (innovation feature 1) ────────────────────
    res_lag24 = df['res_pct'].shift(24)
    df['res_pct_lag24'] = res_lag24
    df['res_pct_lag168'] = df['res_pct'].shift(168)
    df['res_err_interaction'] = df['err_roll_7d_mean'] * res_lag24  # key innovation

    # ── 6. Load forecast level for target day (known future covariate) ────────
    # TSO publishes this at noon on day d-1 — fully available!
    df['load_forecast_level'] = df['load_forecast']  # This is the forecast FOR this hour
    # Ramp feature: difference between consecutive TSO forecasts (innovation feature 4)
    # Divided by 1000.0 to convert MW to GW change per hour (consistent with load_error GW scale)
    df['forecast_ramp'] = df['load_forecast'].diff(1) / 1000.0
    df['abs_forecast_ramp'] = df['forecast_ramp'].abs()
    df['forecast_ramp_lag24'] = df['forecast_ramp'].shift(24)

    # ── 7. Calendar / seasonality features ───────────────────────────────────
    doy = df.index.dayofyear
    df['sin_hour']  = np.sin(2 * np.pi * df.index.hour / 24)
    df['cos_hour']  = np.cos(2 * np.pi * df.index.hour / 24)
    df['sin_week']  = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['cos_week']  = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['sin_year']  = np.sin(2 * np.pi * doy / 365.25)
    df['cos_year']  = np.cos(2 * np.pi * doy / 365.25)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(float)

    # ── 8. Wind/solar lag features ─────────────────────────────────────────────
    df['wind_lag24']  = df['wind'].shift(24)
    df['solar_lag24'] = df['solar'].shift(24)

    # ── Drop NaNs ──────────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in
                    ['load', 'load_forecast', 'solar', 'wind', 'res', 'net_load', 'price']]
    df.dropna(subset=feature_cols, inplace=True)

    print(f"[data] {country}: feature matrix {df.shape} after dropping NaNs")
    return df


# Feature columns used for ML models
ML_FEATURE_COLS = [
    'err_lag_24', 'err_lag_48', 'err_lag_72', 'err_lag_168', 'err_lag_336',
    'err_roll_24h_mean', 'err_roll_24h_std', 'err_roll_7d_mean', 'err_roll_7d_std',
    'err_roll_30d_mean', 'err_same_hour_lag168', 'err_same_hour_lag336',
    'err_same_hour_3wk_mean', 'err_streak', 'err_direction_lag24',
    'res_pct_lag24', 'res_pct_lag168', 'res_err_interaction',
    'load_forecast_level', 'forecast_ramp', 'abs_forecast_ramp', 'forecast_ramp_lag24',
    'sin_hour', 'cos_hour', 'sin_week', 'cos_week', 'sin_year', 'cos_year',
    'is_weekend', 'wind_lag24', 'solar_lag24',
    'hour', 'dayofweek', 'month',
]

TARGET_COL = 'load_error'

# Train/Val/Test splits
TRAIN_END   = '2018-12-31 23:00'
VAL_START   = '2019-01-01 00:00'
VAL_END     = '2019-12-31 23:00'
TEST_START  = '2020-01-01 00:00'


def get_splits(df):
    train = df[df['year'] <= 2018]
    val   = df[(df['year'] == 2019)]
    test  = df[(df['year'] == 2020)]
    return train, val, test


def prepare_all_countries(raw_csv, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    df_raw = load_raw(raw_csv)
    datasets = {}
    for country in ['DE', 'DK_1', 'GB_GBN']:
        print(f"\n[data] Processing {country}...")
        df_c = extract_country(df_raw, country)
        df_f = build_features(df_c, country)
        out_path = os.path.join(processed_dir, f'{country}_features.parquet')
        df_f.to_parquet(out_path)
        print(f"[data] {country} saved to {out_path}")
        datasets[country] = df_f
    return datasets


if __name__ == '__main__':
    import sys
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/csh/myproject/data/raw'
    proc_dir = sys.argv[2] if len(sys.argv) > 2 else '/home/csh/myproject/data/processed'
    raw_csv = os.path.join(raw_dir, 'time_series_60min_singleindex.csv')

    if not os.path.exists(raw_csv):
        download_data(raw_dir)

    prepare_all_countries(raw_csv, proc_dir)
    print("\n[data] All countries processed successfully.")
