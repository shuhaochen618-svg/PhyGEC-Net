import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import core DML functions from analysis_dml
from analysis_dml import (
    load_and_prepare_country_data,
    compute_dml_residuals,
    estimate_local_causal_effect
)

np.random.seed(42)

def run_covid_exclusion(csv_path, countries, z_grid):
    print("\n=========================================")
    # Running COVID exclusion analysis (2015-2019 data only)
    print("ROBUSTNESS: COVID-19 Exclusion (2015-2019)")
    print("=========================================")
    results = {}
    for country in countries:
        df, core_cols = load_and_prepare_country_data(csv_path, country)
        
        # Filter to pre-2020 only
        df_pre = df[df['year'] < 2020]
        print(f"  Filtering {country} to pre-2020: {len(df_pre)} observations (dropped {len(df) - len(df_pre)} rows)")
        
        y_col = 'delta_price'
        d_col = 'load_error'
        z_col = 'res_pct'
        exclude = {y_col, d_col, z_col, 'price', 'load', 'solar', 'wind', 'price_median_7d'}
        x_cols = [c for c in core_cols if c not in exclude]
        
        df_res = compute_dml_residuals(df_pre, y_col, d_col, z_col, x_cols, n_splits=5)
        res_curve = estimate_local_causal_effect(df_res, z_grid, bandwidth=0.06)
        
        out_path = os.path.join("data", f"robustness_covid_excl_{country}.csv")
        res_curve.to_csv(out_path, index=False)
        print(f"  COVID exclusion curve saved to {out_path}")
        results[country] = res_curve
    return results

def run_placebo_test(csv_path, countries, z_grid):
    print("\n=========================================")
    # Running Placebo analysis by permuting the load forecast errors
    print("ROBUSTNESS: Placebo Test (Permuted Load Error)")
    print("=========================================")
    results = {}
    for country in countries:
        df, core_cols = load_and_prepare_country_data(csv_path, country)
        
        # Permute treatment variable (load error)
        print(f"  Permuting load error for {country} to break causal link...")
        df['load_error'] = np.random.permutation(df['load_error'].values)
        
        y_col = 'delta_price'
        d_col = 'load_error'
        z_col = 'res_pct'
        exclude = {y_col, d_col, z_col, 'price', 'load', 'solar', 'wind', 'price_median_7d'}
        x_cols = [c for c in core_cols if c not in exclude]
        
        df_res = compute_dml_residuals(df, y_col, d_col, z_col, x_cols, n_splits=5)
        res_curve = estimate_local_causal_effect(df_res, z_grid, bandwidth=0.06)
        
        out_path = os.path.join("data", f"robustness_placebo_{country}.csv")
        res_curve.to_csv(out_path, index=False)
        print(f"  Placebo curve saved to {out_path}")
        results[country] = res_curve
    return results

def run_bandwidth_sensitivity(countries, z_grid, bandwidths=[0.04, 0.06, 0.10, 0.15]):
    print("\n=========================================")
    # Running bandwidth sensitivity using pre-calculated residuals
    print("ROBUSTNESS: Bandwidth Sensitivity")
    print("=========================================")
    results = {c: {} for c in countries}
    
    for country in countries:
        res_data_path = os.path.join("data", f"{country}_dml_residuals.parquet")
        if not os.path.exists(res_data_path):
            print(f"  Residual file {res_data_path} not found. Skip.")
            continue
            
        print(f"  Loading pre-calculated residuals for {country} from {res_data_path}...")
        df_res = pd.read_parquet(res_data_path)
        
        for h in bandwidths:
            print(f"  Estimating with bandwidth h = {h:.2f}...")
            res_curve = estimate_local_causal_effect(df_res, z_grid, bandwidth=h)
            out_path = os.path.join("data", f"robustness_bandwidth_{h:.2f}_{country}.csv")
            res_curve.to_csv(out_path, index=False)
            results[country][h] = res_curve
            
    return results

def plot_robustness_results(countries, z_grid, covid_res, placebo_res, bandwidth_res):
    print("\nGenerating publication-quality robustness check visualizations...")
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.linestyle': '--'
    })
    
    # 3x3 Plot Matrix
    # Columns: DE, DK_1, GB_GBN
    # Rows: Row 1 = COVID-19 Exclusion, Row 2 = Placebo, Row 3 = Bandwidth Sensitivity
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    
    country_labels = {
        'DE': 'Germany (DE)',
        'DK_1': 'Denmark West (DK_1)',
        'GB_GBN': 'Great Britain (GB_GBN)'
    }
    
    # Load baseline curves for comparison
    baseline_res = {}
    for country in countries:
        baseline_path = os.path.join("data", f"{country}_causal_curve.csv")
        if os.path.exists(baseline_path):
            baseline_res[country] = pd.read_csv(baseline_path)
        else:
            baseline_res[country] = None

    for col_idx, country in enumerate(countries):
        # 1. Row 1: COVID-19 Exclusion vs Baseline
        ax = axes[0, col_idx]
        base_curve = baseline_res[country]
        cov_curve = covid_res[country]
        
        if base_curve is not None:
            # Baseline (Full Sample)
            valid_base = base_curve.dropna(subset=['theta', 'se'])
            ax.plot(valid_base['res_pct'] * 100, valid_base['theta'], 'k--', label='Full Sample (2015-2020)', alpha=0.7)
            
        valid_cov = cov_curve.dropna(subset=['theta', 'se'])
        ax.plot(valid_cov['res_pct'] * 100, valid_cov['theta'], 'b-', lw=2, label='COVID Excluded (2015-2019)')
        ax.fill_between(valid_cov['res_pct'] * 100, 
                        valid_cov['theta'] - 1.96 * valid_cov['se'], 
                        valid_cov['theta'] + 1.96 * valid_cov['se'], 
                        color='blue', alpha=0.1)
        ax.axhline(0, color='gray', lw=0.8, ls='-')
        ax.set_title(f"{country_labels[country]}\nCOVID-19 Subsample Exclusion", fontsize=11, fontweight='semibold')
        if col_idx == 0:
            ax.set_ylabel(r'$\theta(Z)$' + '\n(Price Deviation €/MWh per GW Error)', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        
        # 2. Row 2: Placebo Test
        ax = axes[1, col_idx]
        plac_curve = placebo_res[country]
        valid_plac = plac_curve.dropna(subset=['theta', 'se'])
        
        ax.plot(valid_plac['res_pct'] * 100, valid_plac['theta'], 'r-', lw=2, label='Placebo (Shuffled D)')
        ax.fill_between(valid_plac['res_pct'] * 100, 
                        valid_plac['theta'] - 1.96 * valid_plac['se'], 
                        valid_plac['theta'] + 1.96 * valid_plac['se'], 
                        color='red', alpha=0.1)
        ax.axhline(0, color='gray', lw=0.8, ls='-')
        ax.set_title("Placebo Treatment Test", fontsize=11, fontweight='semibold')
        if col_idx == 0:
            ax.set_ylabel(r'$\theta(Z)$' + '\n(Price Deviation €/MWh per GW Error)', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        
        # 3. Row 3: Bandwidth Sensitivity
        ax = axes[2, col_idx]
        bw_dict = bandwidth_res[country]
        styles = {0.04: ':', 0.06: '-', 0.10: '--', 0.15: '-.'}
        colors = {0.04: '#ff7f0e', 0.06: '#1f77b4', 0.10: '#2ca02c', 0.15: '#9467bd'}
        
        for h, curve in bw_dict.items():
            valid_curve = curve.dropna(subset=['theta', 'se'])
            ax.plot(valid_curve['res_pct'] * 100, valid_curve['theta'], 
                    color=colors[h], ls=styles[h], lw=1.8, label=f'h = {h:.2f}')
            
        ax.axhline(0, color='gray', lw=0.8, ls='-')
        ax.set_title("Bandwidth Sensitivity Test", fontsize=11, fontweight='semibold')
        ax.set_xlabel('Renewable Penetration Z (%)', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(r'$\theta(Z)$' + '\n(Price Deviation €/MWh per GW Error)', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout()
    out_fig_path = os.path.join("figures", "dml_robustness_checks.png")
    plt.savefig(out_fig_path, dpi=300, bbox_inches='tight')
    print(f"\nRobustness figures saved to {out_fig_path}")
    plt.close(fig)

def main():
    csv_path = os.path.join("data", "time_series_60min_singleindex.csv")
    countries = ['DE', 'DK_1', 'GB_GBN']
    z_grid = np.linspace(0.02, 0.75, 40)
    
    # 1. Run COVID-19 Exclusion
    covid_res = run_covid_exclusion(csv_path, countries, z_grid)
    
    # 2. Run Placebo Test
    placebo_res = run_placebo_test(csv_path, countries, z_grid)
    
    # 3. Run Bandwidth Sensitivity
    bandwidth_res = run_bandwidth_sensitivity(countries, z_grid, bandwidths=[0.04, 0.06, 0.10, 0.15])
    
    # 4. Plot results
    plot_robustness_results(countries, z_grid, covid_res, placebo_res, bandwidth_res)
    
    print("\nRobustness checks workflow completed successfully.")

if __name__ == "__main__":
    main()
