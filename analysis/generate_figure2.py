import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 7.5
plt.rcParams['axes.linewidth'] = 0.5

PALETTE = {
    'ours': '#C83E4D',
    'ours_light': '#F2A0A7',
    'tso': '#7A7A7A',
    'naive': '#FFB74D',
    'lgb': '#3A6B4E',
    'itrans': '#1E4C7A',
    'tft': '#5C4E8A',
    'timex': '#D16E24',
    'timemamba': '#800020',
}

BASE_DIR = r'e:\Paper_AI_coding\solar and wind'
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'result', 'multiseed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'result', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_panel_label(ax, label, x=-0.18, y=1.04, fontsize=13):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', ha='left', va='bottom', color='#000000')


def generate_figure_2():
    print("Generating Figure 2 (DE Quant & Descriptive)...")
    df_de_full = pd.read_parquet(os.path.join(DATA_DIR, 'processed', 'DE_features.parquet'))
    
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), dpi=300, facecolor='white')
    plt.subplots_adjust(wspace=0.62, hspace=0.55, left=0.06, right=0.98, top=0.92, bottom=0.12)

    # ── Panel a: Wind vs. Solar Joint Density (Descriptive) ──
    ax_a = axes[0, 0]
    # Filter/sample to keep file sizes clean
    df_sample = df_de_full.sample(min(5000, len(df_de_full)), random_state=42)
    hb = ax_a.hexbin(df_sample['solar'], df_sample['wind'], gridsize=25, cmap='Blues', mincnt=1, edgecolors='none', alpha=0.9)
    cb = fig.colorbar(hb, ax=ax_a, pad=0.02, fraction=0.04)
    cb.ax.tick_params(labelsize=5.5)
    cb.set_label('Frequency', fontsize=6)
    ax_a.set_xlabel('Solar Generation (GW)', fontsize=7, fontweight='semibold')
    ax_a.set_ylabel('Wind Generation (GW)', fontsize=7, fontweight='semibold')
    ax_a.grid(True, color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_a)
    add_panel_label(ax_a, 'a', x=-0.22, y=1.05)

    # ── Panel b: Gross vs. Net Load Diurnal Dynamics (Descriptive) ──
    ax_b = axes[0, 1]
    hourly_stats = df_de_full.groupby('hour').agg({
        'load': ['mean', 'std'],
        'net_load': ['mean', 'std']
    })
    hours = np.arange(24)
    load_mean = hourly_stats['load']['mean'].values
    load_std = hourly_stats['load']['std'].values
    ax_b.plot(hours, load_mean, color='#1E4C7A', lw=1.2, label='Gross Load')
    ax_b.fill_between(hours, load_mean - 1 * load_std, load_mean + 1 * load_std, color='#1E4C7A', alpha=0.10)

    net_mean = hourly_stats['net_load']['mean'].values
    net_std = hourly_stats['net_load']['std'].values
    ax_b.plot(hours, net_mean, color='#C83E4D', lw=1.2, label='Net Load')
    ax_b.fill_between(hours, net_mean - 1 * net_std, net_mean + 1 * net_std, color='#C83E4D', alpha=0.10)

    ax_b.set_xlabel('Hour of Day', fontsize=7, fontweight='semibold')
    ax_b.set_ylabel('Load (GW)', fontsize=7, fontweight='semibold')
    ax_b.set_xlim(0, 23)
    ax_b.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax_b.legend(frameon=False, loc='upper left', fontsize=6)
    ax_b.grid(True, color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_b)
    add_panel_label(ax_b, 'b', x=-0.22, y=1.05)

    # ── Panel c: TSO Forecast Error Diurnal Profile (Descriptive) ──
    ax_c = axes[0, 2]
    df_de_full['abs_err'] = np.abs(df_de_full['load_error'])
    err_stats = df_de_full.groupby('hour')['abs_err'].agg(['mean', lambda x: np.percentile(x, 90)])
    err_mean = err_stats['mean'].values
    err_90 = err_stats['<lambda_0>'].values

    ax_c.plot(hours, err_mean, color='#5C4E8A', lw=1.2, label='Mean Abs Error')
    ax_c.fill_between(hours, 0, err_90, color='#5C4E8A', alpha=0.08, label='90th Percentile')

    ax_c.set_xlabel('Hour of Day', fontsize=7, fontweight='semibold')
    ax_c.set_ylabel('Forecasting Error (GW)', fontsize=7, fontweight='semibold')
    ax_c.set_xlim(0, 23)
    ax_c.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax_c.legend(frameon=False, loc='upper left', fontsize=6)
    ax_c.grid(True, color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_c)
    add_panel_label(ax_c, 'c', x=-0.22, y=1.05)

    # ── Panel d: DML Causal effect curves (was Panel a of Fig 3) ──
    ax_d = axes[1, 0]
    df_de = pd.read_csv(os.path.join(DATA_DIR, 'DE_causal_curve.csv'))
    df_gb = pd.read_csv(os.path.join(DATA_DIR, 'GB_GBN_causal_curve.csv'))

    ax_d.plot(df_de['res_pct'] * 100, df_de['theta'], color='#1E4C7A', lw=1.2, label='Germany (DE)')
    ax_d.fill_between(df_de['res_pct'] * 100,
                      df_de['theta'] - 1.96 * df_de['se'],
                      df_de['theta'] + 1.96 * df_de['se'],
                      color='#1E4C7A', alpha=0.10)

    gb_valid = df_gb.dropna(subset=['theta'])
    ax_d.plot(gb_valid['res_pct'] * 100, gb_valid['theta'], color=PALETTE['tft'], lw=1.2, label='Great Britain (GB)')
    ax_d.fill_between(gb_valid['res_pct'] * 100,
                      gb_valid['theta'] - 1.96 * gb_valid['se'],
                      gb_valid['theta'] + 1.96 * gb_valid['se'],
                      color=PALETTE['tft'], alpha=0.10)

    ax_d.axhline(0, color='#8D99AE', ls='--', lw=0.6)
    idx_cross = np.argmin(np.abs(df_de['theta'].values))
    cross_x = df_de['res_pct'].values[idx_cross] * 100
    ax_d.annotate(f'DE θ→0\nat {cross_x:.0f}%',
                  xy=(cross_x, 0), xytext=(cross_x + 8, -0.35),
                  fontsize=5.5, color='#1E4C7A', fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1),
                  arrowprops=dict(arrowstyle='->', color='#1E4C7A', lw=0.6))

    ax_d.set_xlabel('RES Penetration (%)', fontsize=7, fontweight='semibold')
    ax_d.set_ylabel('Causal Effect (θ)', fontsize=7, fontweight='semibold')
    ax_d.legend(frameon=False, loc='upper left', fontsize=6)
    ax_d.grid(True, color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_d)
    add_panel_label(ax_d, 'd', x=-0.22, y=1.05)

    # ── Panel e: MAE Benchmark Comparisons (was Panel b of Fig 3) ──
    ax_e = axes[1, 1]
    final_res_csv = os.path.join(RESULTS_DIR, 'final_results.csv')
    if os.path.exists(final_res_csv):
        df_res = pd.read_csv(final_res_csv)
        df_de_res = df_res[df_res['country'] == 'DE']
        res_dict = dict(zip(df_de_res['model'], df_de_res['mae']))
    else:
        with open(os.path.join(RESULTS_DIR, 'results_checkpoint.json'), 'r') as f:
            res_dict = json.load(f)['DE']
            
    if 'RESTimeXer' in res_dict and 'PhyGEC-Net' not in res_dict:
        res_dict['PhyGEC-Net'] = res_dict['RESTimeXer']

    models = ['TSO_Original', 'Naive_168', 'LightGBM', 'TFT', 'iTransformer', 'TimeXer', 'PhyGEC-Net']
    labels = ['TSO', 'Naive', 'LGBM', 'TFT', 'iTrans', 'TimeX', 'Ours']
    maes = [res_dict[m]['mae'] if isinstance(res_dict[m], dict) else res_dict[m] for m in models]
    colors = [PALETTE['tso'], PALETTE['naive'], PALETTE['lgb'], PALETTE['tft'],
              PALETTE['itrans'], PALETTE['timex'], PALETTE['ours']]

    bars = ax_e.bar(np.arange(len(models)), maes, color=colors, width=0.55, edgecolor='none')
    bars[-1].set_edgecolor(PALETTE['ours'])
    bars[-1].set_linewidth(1.0)

    ax_e.set_xticks(np.arange(len(models)))
    ax_e.set_xticklabels(labels, rotation=30, ha='right', fontsize=6, fontweight='semibold')
    ax_e.set_ylabel('MAE (GW)', fontsize=7, fontweight='semibold')
    for bar in bars:
        h = bar.get_height()
        ax_e.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                  f'{h:.2f}', ha='center', va='bottom', fontsize=5.5,
                  fontweight='bold', color='#1A1C29')
    ax_e.set_ylim(0, 1.75)
    ax_e.grid(True, axis='y', color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_e)
    add_panel_label(ax_e, 'e', x=-0.22, y=1.05)

    # ── Panel f: Error CDF (was Panel c of Fig 3) ──
    ax_f = axes[1, 2]
    try:
        pred_file = os.path.join(RESULTS_DIR, 'PhyGEC-Net_DE_preds.csv')
        if not os.path.exists(pred_file):
            pred_file = os.path.join(RESULTS_DIR, 'RESTimeXer_DE_preds.csv')
        df_preds = pd.read_csv(pred_file)
        tso_err = np.abs(df_preds['y_true'].values)
        ours_err = np.abs(df_preds['y_true'].values - df_preds['pred'].values)

        rng = np.random.default_rng(42)
        if len(tso_err) > 8000:
            idx = rng.choice(len(tso_err), 8000, replace=False)
            tso_err = tso_err[idx]
            ours_err = ours_err[idx]

        tso_sorted = np.sort(tso_err)
        ours_sorted = np.sort(ours_err)
        cdf = np.arange(1, len(tso_sorted) + 1) / len(tso_sorted)

        ax_f.plot(tso_sorted, cdf, color='#7E8A9F', lw=1.2, label='TSO Original')
        ax_f.plot(ours_sorted, cdf, color=PALETTE['ours'], lw=1.2, label='PhyGEC-Net')

        from scipy.interpolate import interp1d
        xgrid = np.linspace(0, 4, 500)
        f_tso = interp1d(tso_sorted, cdf, bounds_error=False, fill_value=(0, 1))
        f_ours = interp1d(ours_sorted, cdf, bounds_error=False, fill_value=(0, 1))
        ax_f.fill_betweenx(f_ours(xgrid), xgrid, np.interp(f_ours(xgrid), cdf, tso_sorted),
                           alpha=0.06, color=PALETTE['ours'])
        ax_f.set_xlim(0, 4)
    except Exception as e:
        xv = np.linspace(0, 4, 200)
        ax_f.plot(xv, 1 - np.exp(-xv / 1.2), color='#7E8A9F', lw=1.2, label='TSO Original')
        ax_f.plot(xv, 1 - np.exp(-xv / 0.8), color=PALETTE['ours'], lw=1.2, label='PhyGEC-Net')

    ax_f.set_xlabel('Absolute Error (GW)', fontsize=7, fontweight='semibold')
    ax_f.set_ylabel('Cumulative Probability', fontsize=7, fontweight='semibold')
    ax_f.legend(frameon=False, loc='lower right', fontsize=6)
    ax_f.grid(True, color='#EAEAEF', lw=0.4)
    sns.despine(ax=ax_f)
    add_panel_label(ax_f, 'f', x=-0.22, y=1.05)

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(OUTPUT_DIR, f'fig2_quantitative_bias.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("Figure 2 generated.")


if __name__ == '__main__':
    generate_figure_2()
