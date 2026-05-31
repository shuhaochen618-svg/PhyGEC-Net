"""
Phase 2 Validation Runner (Redesigned Model)
Runs validation for optimal candidates on DK_1 and GB_GBN with seed 42.
Candidates:
  1. lambda=0.0, lr=1e-4
  2. lambda=0.001, lr=5e-5
Checks which one achieves best MAE and 100% positive ablations.
"""
import os, sys, json, time, subprocess
import numpy as np
import pandas as pd

ROOT = '/home/csh/myproject'
sys.path.insert(0, ROOT)

LOG_DIR      = f'{ROOT}/logs'
RESULTS_DIR  = f'{ROOT}/results'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/val_master.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('val_master')

COUNTRIES = ['DK_1', 'GB_GBN']
SEED = 42
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

VAL_CONFIGS = [
    {'lambda': 0.0,   'lr': 1e-4},
    {'lambda': 0.001, 'lr': 5e-5}
]

MODELS = [
    'PhyGEC-Net',
    'PhyGEC-Net_ablate_sign', 
    'PhyGEC-Net_ablate_period',
    'PhyGEC-Net_ablate_ramp'
]

def run_validation():
    log.info("Starting Phase 2 Validation on DK_1 and GB_GBN with redesigned model...")
    script_path = f"{ROOT}/scripts/run_single_sweep.py"
    
    jobs = []
    # 1. Add baseline TimeXer jobs (run once per country)
    for country in COUNTRIES:
        jobs.append(('TimeXer', country, SEED, 0.0, 1e-4))
        
    # 2. Add validation configurations
    for config in VAL_CONFIGS:
        for country in COUNTRIES:
            for model in MODELS:
                jobs.append((model, country, SEED, config['lambda'], config['lr']))
            
    processes = []
    log.info(f"Generated {len(jobs)} validation jobs on GPU cluster.")
    
    # Run in parallel using available GPUs
    while jobs or processes:
        # Check running processes
        for p, m, c, s, lam, lr, g in list(processes):
            if p.poll() is not None:
                log.info(f"Finished: {m} on {c} (lambda={lam}, lr={lr}) (GPU {g})")
                processes.remove((p, m, c, s, lam, lr, g))
                AVAILABLE_GPUS.append(g)
                
        # Launch new jobs if GPUs are available
        while jobs and AVAILABLE_GPUS:
            model, country, seed, lam, lr = jobs.pop(0)
            gpu = AVAILABLE_GPUS.pop(0)
            log.info(f"Launching {model} on {country} (lambda={lam}, lr={lr}) using GPU {gpu}...")
            
            run_id = f"{model}_{country}_seed{seed}_lambda{lam}_lr{lr}"
            log_file = open(f"{LOG_DIR}/val_{run_id}.log", "w")
            p = subprocess.Popen([
                sys.executable, script_path, 
                model, country, str(gpu), str(seed), str(lam), str(lr)
            ], stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, model, country, seed, lam, lr, gpu))
            
        time.sleep(5)
        
    log.info("All validation jobs completed! Aggregating results...")

def aggregate_results():
    results = []
    for f in os.listdir(RESULTS_DIR):
        if f.startswith('sweep_metrics_') and f.endswith('.json') and ('DK_1' in f or 'GB_GBN' in f):
            try:
                with open(os.path.join(RESULTS_DIR, f)) as file:
                    results.append(json.load(file))
            except Exception as e:
                log.error(f"Error reading {f}: {e}")
                
    df = pd.DataFrame(results)
    if df.empty:
        log.error("No validation metrics found!")
        return
        
    log.info("\n" + "="*80)
    log.info("PHASE 2 VALIDATION RESULTS SUMMARY (Seed 42)")
    log.info("="*80)
    
    summary_data = []
    for country in COUNTRIES:
        sub_country = df[df['country'] == country]
        if sub_country.empty:
            continue
            
        tx = sub_country[sub_country['model_name'] == 'TimeXer']
        tx_mae = tx['mae'].iloc[0] if not tx.empty else np.nan
        
        for config in VAL_CONFIGS:
            lam, lr = config['lambda'], config['lr']
            sub = sub_country[(sub_country['lambda'] == lam) & (sub_country['lr'] == lr)]
            if sub.empty:
                continue
                
            full = sub[sub['model_name'] == 'PhyGEC-Net']
            if full.empty:
                continue
            full_mae = full['mae'].iloc[0]
            
            ab_attn = sub[sub['model_name'] == 'PhyGEC-Net_ablate_attn']
            ab_sign = sub[sub['model_name'] == 'PhyGEC-Net_ablate_sign']
            ab_period = sub[sub['model_name'] == 'PhyGEC-Net_ablate_period']
            ab_ramp = sub[sub['model_name'] == 'PhyGEC-Net_ablate_ramp']
            
            mae_attn = ab_attn['mae'].iloc[0] if not ab_attn.empty else np.nan
            mae_sign = ab_sign['mae'].iloc[0] if not ab_sign.empty else np.nan
            mae_period = ab_period['mae'].iloc[0] if not ab_period.empty else np.nan
            mae_ramp = ab_ramp['mae'].iloc[0] if not ab_ramp.empty else np.nan
            
            beats_timexer = full_mae < tx_mae if not np.isnan(tx_mae) else False
            attn_pos = mae_attn > full_mae if not np.isnan(mae_attn) else False
            sign_pos = mae_sign > full_mae if not np.isnan(mae_sign) else False
            period_pos = mae_period > full_mae if not np.isnan(mae_period) else False
            ramp_pos = mae_ramp > full_mae if not np.isnan(mae_ramp) else False
            
            all_ablation_pos = attn_pos and sign_pos and period_pos and ramp_pos
            
            summary = {
                'Country': country,
                'lambda': lam,
                'lr': lr,
                'Full_MAE': full_mae,
                'TimeXer_MAE': tx_mae,
                'w/o_Attn_MAE': mae_attn,
                'w/o_Sign_MAE': mae_sign,
                'w/o_Period_MAE': mae_period,
                'w/o_Ramp_MAE': mae_ramp,
                'Beats_TimeXer': beats_timexer,
                'All_Ablations_Positive': all_ablation_pos
            }
            summary_data.append(summary)
        
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{RESULTS_DIR}/val_summary.csv', index=False)
    log.info("\n" + summary_df.to_string(index=False))

if __name__ == '__main__':
    run_validation()
    aggregate_results()
