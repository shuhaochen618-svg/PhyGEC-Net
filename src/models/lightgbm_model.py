"""
LightGBM Baseline for TSO Forecast Error Correction
Uses the innovation feature set from feature engineering
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os, pickle


class LightGBMErrorCorrector:
    """
    Single unified LightGBM model that uses hour as a feature.
    Trains on all hours jointly (more data, better generalization).
    """
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.feature_names = None

    def get_params(self):
        return {
            'objective':        'regression',
            'metric':           'mae',
            'n_estimators':     1000,
            'learning_rate':    0.02,
            'num_leaves':       63,
            'max_depth':        -1,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'reg_alpha':        0.1,
            'reg_lambda':       0.1,
            'verbose':          -1,
            'n_jobs':           4,
            'random_state':     42,
        }

    def fit(self, X_train, y_train, X_val, y_val):
        params = self.get_params()
        self.feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else None

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        # Convert to lgb.train for early stopping support
        params_train = {k: v for k, v in params.items() if k not in ['n_estimators']}
        self.model = lgb.train(
            params_train,
            dtrain,
            num_boost_round=params['n_estimators'],
            valid_sets=[dval],
            callbacks=callbacks,
        )
        print(f"[LightGBM] Best iteration: {self.model.best_iteration}")

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def feature_importance(self):
        """Return SHAP-ready feature importance."""
        if self.model is None:
            return None
        imp = pd.Series(
            self.model.feature_importance(importance_type='gain'),
            index=self.feature_names or [f'f{i}' for i in range(len(self.model.feature_importance()))]
        ).sort_values(ascending=False)
        return imp

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"[LightGBM] Model saved to {path}")

    def load(self, path):
        self.model = lgb.Booster(model_file=path)


class NaiveBaseline:
    """Naive baselines: Naive-168 and Naive-24."""
    def predict_naive168(self, error_series):
        """Predict next 24h error using same 24h from last week."""
        return error_series.shift(168)

    def predict_naive24(self, error_series):
        """Predict next 24h error using same 24h from yesterday."""
        return error_series.shift(24)

    def predict_zero(self, error_series):
        """Zero prediction = TSO original forecast (no correction)."""
        return pd.Series(0.0, index=error_series.index)


def get_ml_features(df, feature_cols):
    """Extract ML feature matrix and target."""
    from src.data_processor import ML_FEATURE_COLS, TARGET_COL
    available_features = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[LightGBM] Warning: missing features {missing[:5]}...")
    X = df[available_features].fillna(0)
    y = df[TARGET_COL]
    return X, y
