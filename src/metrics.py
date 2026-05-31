"""
Evaluation Metrics for TSO Forecast Error Correction
MAE, RMSE, Skill Score, Diebold-Mariano Test
"""
import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100

def skill_score(mae_model, mae_baseline):
    """Skill score: fraction of error reduction vs. baseline. Positive = better."""
    return 1.0 - mae_model / mae_baseline

def diebold_mariano_test(y_true, pred1, pred2, h=24, power=1):
    """
    Diebold-Mariano test for equal forecast accuracy.
    H0: the two forecasts have equal accuracy.
    Returns: (dm_stat, p_value)
    power=1 -> MAE loss differential, power=2 -> MSE loss differential
    """
    e1 = (y_true - pred1) ** power if power > 1 else np.abs(y_true - pred1)
    e2 = (y_true - pred2) ** power if power > 1 else np.abs(y_true - pred2)
    d  = e1 - e2  # positive means pred1 is worse

    T = len(d)
    d_bar = np.mean(d)

    # Newey-West HAC variance with h-1 lags
    gamma0 = np.var(d, ddof=0)
    hac_var = gamma0
    for k in range(1, h):
        w = 1 - k / h
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        hac_var += 2 * w * gamma_k
    hac_var = max(hac_var, 1e-12)

    dm_stat = d_bar / np.sqrt(hac_var / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def compute_all_metrics(y_true, y_pred, baseline_naive168=None, prefix=''):
    """Compute full metric suite."""
    m = {
        f'{prefix}mae':  mae(y_true, y_pred),
        f'{prefix}rmse': rmse(y_true, y_pred),
        f'{prefix}mape': mape(y_true, y_pred),
    }
    if baseline_naive168 is not None:
        m[f'{prefix}skill_score_vs_naive168'] = skill_score(m[f'{prefix}mae'], mae(y_true, baseline_naive168))
        dm_stat, dm_p = diebold_mariano_test(y_true, y_pred, baseline_naive168)
        m[f'{prefix}dm_stat'] = dm_stat
        m[f'{prefix}dm_pval'] = dm_p
    return m


def tso_benchmark_metrics(df_test, correction_pred_gw):
    """
    Compare TSO-corrected forecast vs. TSO original.
    TSO original: predict error = 0 (i.e., no correction).
    """
    y_true = df_test['load_error'].values[:len(correction_pred_gw)]

    # TSO original: zero correction = predicting error is 0
    pred_zero = np.zeros_like(y_true)

    # Naive-168: use same hour from last week
    naive168 = df_test['err_lag_168'].values[:len(correction_pred_gw)]
    naive24  = df_test['err_lag_24'].values[:len(correction_pred_gw)]

    results = {}
    results['TSO_Original']  = compute_all_metrics(y_true, pred_zero)
    results['Naive_168']     = compute_all_metrics(y_true, naive168, pred_zero)
    results['Naive_24']      = compute_all_metrics(y_true, naive24, pred_zero)
    results['Our_Model']     = compute_all_metrics(y_true, correction_pred_gw, pred_zero,
                                                    prefix='')
    # DM test: our model vs naive-168
    dm_stat, dm_p = diebold_mariano_test(y_true, correction_pred_gw, naive168)
    results['Our_Model']['dm_vs_naive168_stat'] = dm_stat
    results['Our_Model']['dm_vs_naive168_pval'] = dm_p

    # Skill score
    results['Our_Model']['skill_vs_tso'] = skill_score(
        results['Our_Model']['mae'], results['TSO_Original']['mae'])

    return results, y_true
