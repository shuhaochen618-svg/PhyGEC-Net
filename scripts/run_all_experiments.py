"""
Master Experiment Runner - Parallel execution on multiple GPUs
Orchestrates all experiments: LightGBM -> Deep Learning (iTransformer, TFT, TimeXer)
Reports progress to logs/master.log
"""
import os, sys, json, time, traceback, subprocess
import numpy as np
import pandas as pd

ROOT = '/home/csh/myproject'
sys.path.insert(0, ROOT)

LOG_DIR      = f'{ROOT}/logs'
RESULTS_DIR  = f'{ROOT}/results'
FIGURES_DIR  = f'{ROOT}/figures'
DATA_RAW     = f'{ROOT}/data/raw'
DATA_PROC    = f'{ROOT}/data/processed'

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROC, exist_ok=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/master.log'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('master')

COUNTRIES = ['DE', 'DK_1', 'GB_GBN']
MODELS = [
    'TFT', 'iTransformer', 'TimeXer', 'PhyGEC-Net',
    'PhyGEC-Net_ablate_sign', 
    'PhyGEC-Net_ablate_period', 'PhyGEC-Net_ablate_ramp'
]
AVAILABLE_GPUS = [1, 2, 3, 5, 6, 7]

def step_status(step_name, status, details=''):
    msg = f"{'='*60}\n[STEP] {step_name} | {status}\n{details}\n{'='*60}"
    log.info(msg)
    with open(f'{LOG_DIR}/status.json', 'w') as f:
        json.dump({'step': step_name, 'status': status,
                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f)

def phase_data():
    step_status('Phase 0: Data', 'STARTING')
    from src.data_processor import download_data, prepare_all_countries
    raw_csv = os.path.join(DATA_RAW, 'time_series_60min_singleindex.csv')
    if not os.path.exists(raw_csv):
        log.info("Downloading OPSD data from server...")
        download_data(DATA_RAW)
    else:
        log.info(f"Raw CSV found: {raw_csv}")

    proc_files = [os.path.join(DATA_PROC, f'{c}_features.parquet') for c in COUNTRIES]
    if all(os.path.exists(f) for f in proc_files):
        log.info("Processed files already exist.")
        datasets = {c: pd.read_parquet(os.path.join(DATA_PROC, f'{c}_features.parquet'))
                    for c in COUNTRIES}
    else:
        datasets = prepare_all_countries(raw_csv, DATA_PROC)

    step_status('Phase 0: Data', 'DONE')
    return datasets

def phase_lightgbm(datasets):
    step_status('Phase 1: LightGBM', 'STARTING')
    import lightgbm as lgb
    from src.data_processor import ML_FEATURE_COLS, TARGET_COL, get_splits
    from src.metrics import compute_all_metrics, tso_benchmark_metrics

    all_results = {}
    chk = f'{RESULTS_DIR}/results_checkpoint.json'
    if os.path.exists(chk):
        try:
            with open(chk) as f:
                all_results = json.load(f)
        except:
            pass

    for country in COUNTRIES:
        log.info(f"\n[LightGBM] Training for {country}...")
        df = datasets[country]
        train, val, test = get_splits(df)

        avail = [c for c in ML_FEATURE_COLS if c in df.columns]
        X_tr, y_tr = train[avail].fillna(0), train[TARGET_COL]
        X_va, y_va = val[avail].fillna(0),   val[TARGET_COL]
        X_te, y_te = test[avail].fillna(0),  test[TARGET_COL]

        from src.models.lightgbm_model import LightGBMErrorCorrector
        model = LightGBMErrorCorrector()
        model.fit(X_tr, y_tr, X_va, y_va)
        model.save(f'{RESULTS_DIR}/lgb_{country}.txt')

        pred_te = model.predict(X_te)
        results, y_true = tso_benchmark_metrics(test, pred_te)
        results['LightGBM'] = compute_all_metrics(y_true, pred_te)

        if country not in all_results:
            all_results[country] = {}
        all_results[country]['LightGBM'] = results['LightGBM']
        all_results[country]['TSO_Original'] = results['TSO_Original']
        all_results[country]['Naive_168'] = results['Naive_168']

        pd.DataFrame({'y_true': y_true, 'lgb_pred': pred_te}).to_csv(
            f'{RESULTS_DIR}/lgb_{country}_preds.csv', index=False)

    with open(chk, 'w') as f:
        json.dump(all_results, f, indent=2)
    step_status('Phase 1: LightGBM', 'DONE')


def write_single_dl_script():
    """Generates a script to run a single DL model for a single country."""
    script_content = f"""
import os, sys, json, torch, importlib, traceback
import pandas as pd
import numpy as np

ROOT = '{ROOT}'
sys.path.insert(0, ROOT)

from src.dataset import ErrorCorrectionDataset
from src.trainer import DeepTrainer, seed_everything
from src.metrics import compute_all_metrics, mae, skill_score
from src.data_processor import get_splits

model_name = sys.argv[1]
country = sys.argv[2]
device_id = sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42

log_dir = f'{{ROOT}}/logs'
res_dir = f'{{ROOT}}/results'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

device = torch.device(f'cuda:{{device_id}}' if torch.cuda.is_available() else 'cpu')

DL_CONFIG = {{
    'seq_len':    168,
    'pred_len':   24,
    'batch_size': 32,
    'max_epochs': 100,
    'patience':   15,
    'lr':         1e-4,
    'weight_decay': 1e-4,
}}

MODEL_CONFIGS = {{
    'PhyGEC-Net': {{
        'DE': {{'d_model': 128, 'n_heads': 4, 'e_layers': 2, 'patch_len': 24, 'stride': 12, 'dropout': 0.15, 'lr': 5e-5, 'weight_decay': 1e-4, 'scale_l1_lambda': 0.001}},
        'DK_1': {{'d_model': 128, 'n_heads': 4, 'e_layers': 2, 'patch_len': 24, 'stride': 12, 'dropout': 0.15, 'lr': 5e-5, 'weight_decay': 1e-4, 'scale_l1_lambda': 0.001}},
        'GB_GBN': {{'d_model': 128, 'n_heads': 4, 'e_layers': 2, 'patch_len': 24, 'stride': 12, 'dropout': 0.15, 'lr': 5e-5, 'weight_decay': 1e-4, 'scale_l1_lambda': 0.001}},
    }},
    'TimeXer': {{'d_model': 128, 'n_heads': 4, 'e_layers': 2, 'patch_len': 24, 'stride': 12, 'dropout': 0.15}},
    'iTransformer': {{'d_model': 128, 'n_heads': 4, 'e_layers': 2, 'dropout': 0.15}},
    'TFT': {{'d_model': 128, 'n_heads': 4, 'dropout': 0.15}},
}}

model_builders = {{
    'iTransformer': ('src.models.itransformer', 'build_itransformer'),
    'TFT':          ('src.models.tft',          'build_tft'),
    'TimeXer':      ('src.models.timexer',       'build_timexer'),
    'PhyGEC-Net':   ('src.models.restimexer',    'build_restimexer'),
    'PhyGEC-Net_ablate_attn': ('src.models.restimexer', 'build_restimexer'),
    'PhyGEC-Net_ablate_sign': ('src.models.restimexer', 'build_restimexer'),
    'PhyGEC-Net_ablate_period': ('src.models.restimexer', 'build_restimexer'),
    'PhyGEC-Net_ablate_ramp': ('src.models.restimexer', 'build_restimexer'),
}}

try:
    print(f"Loading data for {{country}}...")
    df = pd.read_parquet(f'{{ROOT}}/data/processed/{{country}}_features.parquet')
    train_df, val_df, test_df = get_splits(df)

    train_ds = ErrorCorrectionDataset(train_df, seq_len=DL_CONFIG['seq_len'], pred_len=DL_CONFIG['pred_len'])
    val_ds   = ErrorCorrectionDataset(val_df,   seq_len=DL_CONFIG['seq_len'], pred_len=DL_CONFIG['pred_len'], scaler_stats=train_ds.scaler_stats)
    test_ds  = ErrorCorrectionDataset(test_df,  seq_len=DL_CONFIG['seq_len'], pred_len=DL_CONFIG['pred_len'], scaler_stats=train_ds.scaler_stats)

    module_path, builder_fn = model_builders[model_name]
    mod = importlib.import_module(module_path)
    builder = getattr(mod, builder_fn)
    
    base_model_name = 'PhyGEC-Net' if 'PhyGEC-Net' in model_name else model_name
    cfg_entry = MODEL_CONFIGS[base_model_name]
    if isinstance(cfg_entry, dict) and any(c in cfg_entry for c in ['DE', 'DK_1', 'GB_GBN']):
        model_specific = cfg_entry[country]
    else:
        model_specific = cfg_entry
        
    model_cfg = {{**DL_CONFIG, **model_specific, 'n_endog': train_ds.n_endog, 'n_exog': train_ds.n_exog, 'n_future': train_ds.n_future}}
    
    # Feature index mapping setup
    model_cfg['feat_to_idx'] = {{
        'res_pct_lag24': 0,
        'err_same_hour_lag168': 5,
        'err_same_hour_lag336': 6,
        'err_same_hour_lag504': 7,
        'err_streak': 8,
        'forecast_ramp': 1,
        'abs_forecast_ramp': 9
    }}
    
    # Ablation configuration flags
    if 'ablate_attn' in model_name:
        model_cfg['ablate_attention'] = True
    elif 'ablate_sign' in model_name:
        model_cfg['ablate_sign_gating'] = True
    elif 'ablate_period' in model_name:
        model_cfg['ablate_periodicity'] = True
    elif 'ablate_ramp' in model_name:
        model_cfg['ablate_ramp_decoder'] = True
        
    seed_everything(seed)
    model = builder(model_cfg)

    # BUG FIX: pass model_cfg containing customized learning rates and weight decays instead of DL_CONFIG
    trainer = DeepTrainer(model, model_cfg, device=device)
    trainer.fit(train_ds, val_ds, model_name=f'{{model_name}}_{{country}}', log_dir=log_dir, seed=seed)

    preds, targets = trainer.predict(test_ds, train_ds.scaler_stats)
    test_mae = mae(targets, preds)

    # Load results to compute skill score
    chk = f'{{res_dir}}/results_checkpoint.json'
    all_res = {{}}
    if os.path.exists(chk):
        with open(chk, 'r') as f:
            all_res = json.load(f)
    
    tso_mae = all_res.get(country, {{}}).get('TSO_Original', {{}}).get('mae')
    ss = skill_score(test_mae, tso_mae) if tso_mae else None

    if country not in all_res: all_res[country] = {{}}
    all_res[country][model_name] = {{
        'mae': float(test_mae),
        'rmse': float(np.sqrt(np.mean((targets - preds)**2))),
        'skill_vs_tso': float(ss) if ss else None,
    }}
    
    with open(chk, 'w') as f:
        json.dump(all_res, f, indent=2)

    pd.DataFrame({{'y_true': targets.flatten(), 'pred': preds.flatten()}}).to_csv(f'{{res_dir}}/{{model_name}}_{{country}}_preds.csv', index=False)
    torch.save(model.state_dict(), f'{{res_dir}}/{{model_name}}_{{country}}.pt')
    print(f"SUCCESS: {{model_name}} {{country}} MAE={{test_mae:.4f}}")

except Exception as e:
    print(f"FAILED {{model_name}} {{country}}: {{e}}")
    traceback.print_exc()
"""
    script_path = f"{ROOT}/scripts/run_single_dl.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path


def phase_deep_learning_parallel():
    step_status('Phase 2: Deep Learning (Parallel)', 'STARTING')
    script_path = write_single_dl_script()
    
    jobs = []
    # Create all tasks
    for model in MODELS:
        for country in COUNTRIES:
            jobs.append((model, country))
            
    processes = []
    
    # Launch up to len(AVAILABLE_GPUS) jobs at once
    while jobs or processes:
        # Check running processes
        for p, m, c, g in list(processes):
            if p.poll() is not None:
                log.info(f"Finished: {m} on {c} (GPU {g})")
                processes.remove((p, m, c, g))
                AVAILABLE_GPUS.append(g)
                
        # Launch new jobs if GPUs are available
        while jobs and AVAILABLE_GPUS:
            model, country = jobs.pop(0)
            gpu = AVAILABLE_GPUS.pop(0)
            log.info(f"Launching {model} on {country} using GPU {gpu}...")
            
            log_file = open(f"{LOG_DIR}/{model}_{country}.log", "w")
            p = subprocess.Popen([sys.executable, script_path, model, country, str(gpu)], stdout=log_file, stderr=subprocess.STDOUT)
            processes.append((p, model, country, gpu))
            
        time.sleep(10)

    step_status('Phase 2: Deep Learning (Parallel)', 'DONE')

def phase_results():
    step_status('Phase 3: Results & Figures', 'STARTING')
    from src.visualize import generate_all_figures

    chk = f'{RESULTS_DIR}/results_checkpoint.json'
    if os.path.exists(chk):
        with open(chk) as f:
            results = json.load(f)
    else:
        log.error("No results found!")
        return

    # Print summary table
    log.info("\n" + "="*80)
    log.info("FINAL RESULTS SUMMARY")
    log.info("="*80)
    models_order = [
        'TSO_Original', 'Naive_168', 'LightGBM', 'iTransformer', 'TFT', 'TimeXer', 'PhyGEC-Net',
        'PhyGEC-Net_ablate_attn', 'PhyGEC-Net_ablate_sign', 'PhyGEC-Net_ablate_period', 'PhyGEC-Net_ablate_ramp'
    ]
    header = f"{'Model':<25}" + "".join([f" {c:>12}" for c in COUNTRIES])
    log.info(header)
    for m in models_order:
        row = f"{m:<25}"
        for c in COUNTRIES:
            v = results.get(c, {}).get(m, {})
            mae_val = v.get('mae', float('nan'))
            row += f" {mae_val:>12.4f}"
        log.info(row)
    log.info("="*80)

    rows = []
    for c in COUNTRIES:
        for m in models_order:
            v = results.get(c, {}).get(m, {})
            rows.append({'country': c, 'model': m, **v})
    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/final_results.csv', index=False)

    try:
        generate_all_figures(results, RESULTS_DIR, FIGURES_DIR)
    except Exception as e:
        log.error(f"Figure generation failed: {e}")

    step_status('Phase 3: Results & Figures', 'DONE')

if __name__ == '__main__':
    log.info("="*80)
    log.info("TSO FORECAST ERROR CORRECTION - PARALLEL PIPELINE")
    log.info("="*80)
    
    t_start = time.time()
    datasets = phase_data()
    phase_lightgbm(datasets)
    phase_deep_learning_parallel()
    phase_results()
    
    total = (time.time() - t_start) / 60
    log.info(f"\nAll experiments completed in {total:.1f} minutes.")
    step_status('PIPELINE', 'COMPLETE', f'Total time: {total:.1f} min')
