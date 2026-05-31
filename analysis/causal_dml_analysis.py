import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# --- 1. Data Preparation and Preprocessing ---
def load_and_prepare_country_data(csv_path, country):
    print(f"Loading data for {country} from OPSD CSV...")
    df = pd.read_csv(csv_path, parse_dates=['utc_timestamp'])
    df.set_index('utc_timestamp', inplace=True)
    df.index = df.index.tz_localize(None)
    
    # Country-specific column mappings
    configs = {
        'DE': {
            'price': lambda df: df['DE_LU_price_day_ahead'].fillna(df['AT_price_day_ahead']),
            'load': 'DE_load_actual_entsoe_transparency',
            'load_forecast': 'DE_load_forecast_entsoe_transparency',
            'solar': 'DE_solar_generation_actual',
            'wind': lambda df: df['DE_wind_onshore_generation_actual'].fillna(0) + df['DE_wind_offshore_generation_actual'].fillna(0)
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
    
    # Extract raw series
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
        
    # Build core variables
    temp['load_error'] = (temp['load'] - temp['load_forecast']) / 1000.0  # in GW
    temp['res_pct'] = (temp['solar'] + temp['wind']) / temp['load']
    temp['res_pct'] = temp['res_pct'].clip(0, 1)
    
    # 7-day rolling median price as baseline
    temp['price_median_7d'] = temp['price'].rolling(window=168, min_periods=48).median()
    temp['delta_price'] = temp['price'] - temp['price_median_7d']
    
    # Temporal controls
    temp['hour'] = temp.index.hour
    temp['dayofweek'] = temp.index.dayofweek
    temp['month'] = temp.index.month
    temp['year'] = temp.index.year
    
    # Lags controls
    temp['price_lag_24'] = temp['price'].shift(24)
    temp['price_lag_48'] = temp['price'].shift(48)
    temp['price_lag_168'] = temp['price'].shift(168)
    
    temp['load_lag_24'] = temp['load'].shift(24)
    temp['load_lag_48'] = temp['load'].shift(48)
    
    temp['wind_lag_24'] = temp['wind'].shift(24)
    temp['solar_lag_24'] = temp['solar'].shift(24)
    
    # Drop rows with NaNs
    core_cols = [
        'delta_price', 'load_error', 'res_pct', 'load_forecast',
        'hour', 'dayofweek', 'month', 'year',
        'price_lag_24', 'price_lag_48', 'price_lag_168',
        'load_lag_24', 'load_lag_48', 'wind_lag_24', 'solar_lag_24'
    ]
    temp.dropna(subset=core_cols, inplace=True)
    print(f"  Processed {country} successfully. Clean observations: {len(temp)}")
    return temp, core_cols

# --- 2. Double Machine Learning (DML) Nuisance Estimator ---
def compute_dml_residuals(df, y_col, d_col, z_col, x_cols, n_splits=5):
    print(f"Running DML double residualization via {n_splits}-fold weekly block cross-fitting...")
    
    # Create weekly blocks to avoid time-series leakage
    df['week_block'] = df.index.year * 100 + df.index.isocalendar().week
    unique_blocks = df['week_block'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_blocks)
    
    folds = np.array_split(unique_blocks, n_splits)
    
    y_pred = np.zeros(len(df))
    d_pred = np.zeros(len(df))
    
    # Nuisance models use both control variables X and moderator Z
    features = x_cols + [z_col]
    
    for k in range(n_splits):
        val_blocks = folds[k]
        train_blocks = [b for b in unique_blocks if b not in val_blocks]
        
        train_idx = df['week_block'].isin(train_blocks)
        val_idx = df['week_block'].isin(val_blocks)
        
        X_train, y_train, d_train = df.loc[train_idx, features], df.loc[train_idx, y_col], df.loc[train_idx, d_col]
        X_val = df.loc[val_idx, features]
        
        # Fit XGBoost for Y (Outcome nuisance model)
        model_y = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model_y.fit(X_train, y_train)
        y_pred[val_idx.values] = model_y.predict(X_val)
        
        # Fit XGBoost for D (Treatment nuisance model)
        model_d = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        model_d.fit(X_train, d_train)
        d_pred[val_idx.values] = model_d.predict(X_val)
        
        print(f"  Fold {k+1}/{n_splits} complete.")
        
    df['y_residual'] = df[y_col] - y_pred
    df['d_residual'] = df[d_col] - d_pred
    return df

# --- 3. Non-parametric Local Kernel Weighting ---
def estimate_local_causal_effect(df, z_grid, bandwidth=0.05):
    print(f"Estimating local causal effect curves over grid (bandwidth={bandwidth})...")
    theta_list = []
    se_list = []
    ess_list = []
    
    y_res = df['y_residual'].values
    d_res = df['d_residual'].values
    z_val = df['res_pct'].values
    
    for z in z_grid:
        # Gaussian kernel weight calculation
        diff = z_val - z
        weights = np.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
        
        # Effective Sample Size check
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights ** 2)
        ess = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0
        
        if ess < 100:  # Skip boundary values with low local density
            theta_list.append(np.nan)
            se_list.append(np.nan)
            ess_list.append(ess)
            continue
            
        # Closed-form local treatment effect
        num = np.sum(weights * d_res * y_res)
        den = np.sum(weights * d_res ** 2)
        
        if den == 0:
            theta_list.append(np.nan)
            se_list.append(np.nan)
            ess_list.append(ess)
            continue
            
        theta = num / den
        
        # Asymptotic variance (Sandwich formula)
        errors = y_res - theta * d_res
        var_num = np.sum((weights ** 2) * (d_res ** 2) * (errors ** 2))
        var = var_num / (den ** 2)
        se = np.sqrt(var)
        
        theta_list.append(theta)
        se_list.append(se)
        ess_list.append(ess)
        
    res_df = pd.DataFrame({
        'res_pct': z_grid,
        'theta': theta_list,
        'se': se_list,
        'ess': ess_list
    })
    return res_df

# --- 4. Main Workflow Execution ---
def main():
    csv_path = os.path.join("data", "time_series_60min_singleindex.csv")
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    countries = ['DE', 'DK_1', 'GB_GBN']
    results = {}
    
    # Run DML + Local Kernel Regression for each country
    for country in countries:
        print(f"\n=========================================")
        print(f"PROCESS: Causal DML Estimation for {country}")
        print(f"=========================================")
        
        # Load and prep
        df, core_cols = load_and_prepare_country_data(csv_path, country)
        
        # Define variable roles
        y_col = 'delta_price'
        d_col = 'load_error'
        z_col = 'res_pct'
        
        # Exclude target, treatment, and moderator from controls X
        exclude = {y_col, d_col, z_col, 'price', 'load', 'solar', 'wind', 'price_median_7d'}
        x_cols = [c for c in core_cols if c not in exclude]
        
        # 1. DML double residualization
        df_res = compute_dml_residuals(df, y_col, d_col, z_col, x_cols, n_splits=5)
        
        # Save residuals dataset for transparency/replication
        res_data_path = os.path.join("data", f"{country}_dml_residuals.parquet")
        df_res.to_parquet(res_data_path)
        print(f"  Residualized dataset saved to {res_data_path}")
        
        # 2. Local estimation
        # Define grid for Z (DK_1 has higher wind, DE has high solar, GB has wind)
        # We search Z in 0.02 to 0.75 range
        z_grid = np.linspace(0.02, 0.75, 40)
        res_curve = estimate_local_causal_effect(df_res, z_grid, bandwidth=0.06)
        
        # Save curve data
        curve_path = os.path.join("data", f"{country}_causal_curve.csv")
        res_curve.to_csv(curve_path, index=False)
        print(f"  Causal curve data saved to {curve_path}")
        
        results[country] = res_curve
        
    # --- 5. Generate Nature-Caliber Plots ---
    print("\nGenerating publication-quality causal curves...")
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'xtick.direction': 'out',
        'ytick.direction': 'out'
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), sharey=False)
    fig.suptitle("Causal Transmission of TSO Load Forecast Errors to Day-Ahead Prices\nNon-Parametric Causal Curves via Local Double Machine Learning (2015-2020)",
                 fontsize=15, fontweight='bold', y=0.98)
    
    colors = {
        'DE': '#1f77b4',     # Steel Blue
        'DK_1': '#d62728',   # Crimson Red
        'GB_GBN': '#2ca02c'  # Forest Green
    }
    
    titles = {
        'DE': 'Germany (Continental Hub)\nLarge, Interconnected Mixed System',
        'DK_1': 'Denmark West (Jutland)\nSmall, Wind-Saturated System',
        'GB_GBN': 'Great Britain (Island Grid)\nLarge-Scale Decoupled System'
    }
    
    for i, country in enumerate(countries):
        ax = axes[i]
        res = results[country]
        
        # Filter NaNs for plotting
        valid = res.dropna(subset=['theta', 'se'])
        z_val = valid['res_pct'] * 100 # Convert to percentage
        theta = valid['theta']
        ci_lower = theta - 1.96 * valid['se']
        ci_upper = theta + 1.96 * valid['se']
        
        # Plot curve
        ax.plot(z_val, theta, color=colors[country], lw=3, label=r'$\theta(Z)$ Causal Effect')
        ax.fill_between(z_val, ci_lower, ci_upper, color=colors[country], alpha=0.15, label='95% Confidence Interval')
        
        # Add baseline references
        ax.axhline(0, color='black', lw=1.2, ls='-')
        
        # Identify reversal point
        # Find where theta crosses 0 or gets closest to 0
        if len(theta) > 0:
            zero_crossing = None
            for idx in range(len(theta) - 1):
                t1, t2 = theta.iloc[idx], theta.iloc[idx+1]
                z1, z2 = z_val.iloc[idx], z_val.iloc[idx+1]
                if t1 * t2 < 0 or t1 == 0:
                    # Interpolate zero crossing
                    zero_crossing = z1 - t1 * (z2 - z1) / (t2 - t1)
                    break
            
            if zero_crossing is not None:
                ax.axvline(zero_crossing, color='gray', lw=1.2, ls=':', alpha=0.8)
                ax.text(zero_crossing + 1, (theta.max() + theta.min())/2, 
                        f'Reversal Point\n{zero_crossing:.1f}% RES', 
                        fontsize=10, color='dimgray', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.set_title(titles[country], fontsize=13, fontweight='semibold', pad=12)
        ax.set_xlabel('Renewable Penetration Rate Z (%)', fontsize=11, labelpad=8)
        if i == 0:
            ax.set_ylabel('Causal Effect ' + r'$\theta(Z)$' + '\n(Price Deviation €/MWh per GW Load Error)', fontsize=12, labelpad=8)
        else:
            ax.set_ylabel('Causal Effect ' + r'$\theta(Z)$' + ' (€/MWh per GW)', fontsize=11, labelpad=8)
            
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='none')
        
    plt.tight_layout()
    output_fig_path = os.path.join(fig_dir, "dml_causal_curves.png")
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualizations successfully generated and saved to {output_fig_path}")
    plt.close(fig)
    print("DONE: Double Machine Learning mechanism analysis completed successfully.")

if __name__ == "__main__":
    main()
