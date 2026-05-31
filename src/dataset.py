"""
PyTorch Dataset for TSO Forecast Error Correction
Supports TimeXer, iTransformer, TFT input formats
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ErrorCorrectionDataset(Dataset):
    """
    Sliding window dataset for load forecast error correction.
    
    For each sample at time t:
    - x_enc: (seq_len, n_endog)  historical load_error + endogenous features
    - x_exog: (seq_len, n_exog)  historical exogenous features
    - x_mark: (seq_len, n_time)  time features for history window
    - x_future_known: (pred_len, n_future)  known future covariates (load_forecast, calendar)
    - y: (pred_len,)  target load_error to predict
    """
    ENDOG_COLS = ['load_error']  # primary endogenous target series

    EXOG_HIST_COLS = [  # historical exogenous (observable)
        'res_pct_lag24', 'wind_lag24', 'solar_lag24',
        'err_roll_7d_mean', 'err_roll_7d_std',
        'err_same_hour_lag168', 'err_same_hour_lag336', 'err_same_hour_lag504',
        'err_streak',
        'res_err_interaction',
    ]

    FUTURE_KNOWN_COLS = [  # known future at prediction time
        'load_forecast_level', 'forecast_ramp',
        'sin_hour', 'cos_hour', 'sin_week', 'cos_week', 'sin_year', 'cos_year',
        'is_weekend', 'abs_forecast_ramp',
    ]

    TIME_MARK_COLS = ['hour', 'dayofweek', 'month', 'is_weekend']

    def __init__(self, df, seq_len=168, pred_len=24, scaler_stats=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Validate columns
        missing = [c for c in self.ENDOG_COLS + self.EXOG_HIST_COLS + self.FUTURE_KNOWN_COLS
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Store as numpy for efficiency
        self.endog     = df[self.ENDOG_COLS].values.astype(np.float32)
        self.exog_hist = df[self.EXOG_HIST_COLS].values.astype(np.float32)
        self.future_kn = df[self.FUTURE_KNOWN_COLS].values.astype(np.float32)
        self.time_mark = df[self.TIME_MARK_COLS].values.astype(np.float32)
        self.target    = df['load_error'].values.astype(np.float32)

        # Normalization (computed from train if provided, else from this split)
        if scaler_stats is None:
            self.endog_mean  = np.nanmean(self.endog, axis=0)
            self.endog_std   = np.nanstd(self.endog, axis=0) + 1e-8
            self.exog_mean   = np.nanmean(self.exog_hist, axis=0)
            self.exog_std    = np.nanstd(self.exog_hist, axis=0) + 1e-8
            self.fut_mean    = np.nanmean(self.future_kn, axis=0)
            self.fut_std     = np.nanstd(self.future_kn, axis=0) + 1e-8
            self.scaler_stats = {
                'endog_mean': self.endog_mean, 'endog_std': self.endog_std,
                'exog_mean':  self.exog_mean,  'exog_std':  self.exog_std,
                'fut_mean':   self.fut_mean,   'fut_std':   self.fut_std,
            }
        else:
            self.scaler_stats = scaler_stats
            self.endog_mean  = scaler_stats['endog_mean']
            self.endog_std   = scaler_stats['endog_std']
            self.exog_mean   = scaler_stats['exog_mean']
            self.exog_std    = scaler_stats['exog_std']
            self.fut_mean    = scaler_stats['fut_mean']
            self.fut_std     = scaler_stats['fut_std']

        # Normalize
        self.endog     = (self.endog - self.endog_mean) / self.endog_std
        self.exog_hist = (self.exog_hist - self.exog_mean) / self.exog_std
        self.future_kn = (self.future_kn - self.fut_mean) / self.fut_std

        # Number of valid samples
        self.n = len(df) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        s = idx
        e = idx + self.seq_len
        t = e + self.pred_len

        x_enc         = torch.from_numpy(self.endog[s:e])          # (seq_len, 1)
        x_exog        = torch.from_numpy(self.exog_hist[s:e])      # (seq_len, n_exog)
        x_mark        = torch.from_numpy(self.time_mark[s:e])      # (seq_len, 4)
        x_future      = torch.from_numpy(self.future_kn[e:t])      # (pred_len, n_future)
        y_raw         = torch.from_numpy(self.target[e:t])          # (pred_len,) unnormalized
        y_norm        = torch.from_numpy(
            (self.target[e:t] - self.endog_mean[0]) / self.endog_std[0]
        )

        return {
            'x_enc':     x_enc,
            'x_exog':    x_exog,
            'x_mark':    x_mark,
            'x_future':  x_future,
            'y_norm':    y_norm,      # normalized target (for loss)
            'y_raw':     y_raw,       # raw target in GW (for metrics)
        }

    @property
    def n_endog(self):
        return len(self.ENDOG_COLS)

    @property
    def n_exog(self):
        return len(self.EXOG_HIST_COLS)

    @property
    def n_future(self):
        return len(self.FUTURE_KNOWN_COLS)

    @property
    def n_time(self):
        return len(self.TIME_MARK_COLS)
