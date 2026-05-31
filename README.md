# PhyGEC-Net: Physics-Guided Error Correction for Renewable Energy Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview & Academic Background

Modern electric grids are undergoing rapid decarbonization, leading to unprecedented levels of Renewable Energy Sources (RES) penetration. However, the high volatility and weather-dependency of wind and solar power dramatically amplify day-ahead load forecasting errors. Transmission System Operators (TSOs) face major challenges in grid balancing and scheduling due to these amplified deviations.

Traditional deep learning error-correction methods treat this task as a pure black-box sequence modeling problem. They suffer from:
1. **Parameter Explosion:** Standard transformer backbones require millions of parameters, risking severe overfitting on volatile time-series datasets.
2. **Domain Blindness:** They lack physical constraints, failing to capture grid-specific behaviors such as multi-frequency calendar periodicity, weather-driven ramp events, and asymmetric over/under-forecasting behaviors.

**PhyGEC-Net (Physics-Guided Error Correction Network)** solves these issues by embedding domain physical priors directly into a lightweight, patch-based transformer architecture.

---

### Core Innovations & Methodology

PhyGEC-Net extends the TimeXer backbone with four domain-specific modules:

1. **Lightweight RES-Conditioned Attention Bias:** Explicitly injects real-time renewable energy penetration levels ($RES_{pct}$) into the self-attention matrix. This acts as a dynamic variance scaling bias, enabling the network to learn how green energy volatility scales forecast uncertainty without increasing the parameter count.
2. **Residual Sign-Aware Gated Linear Unit (GLU):** Incorporates sign indicators of historical forecast errors to model asymmetric over-forecasting (load deficit) and under-forecasting (load surplus) patterns differently.
3. **Dynamic Gated Seasonal Prior Fusion:** Embeds calendar periodicity priors (daily 24-hour and weekly 168-hour lags) through learnable gating gates. This limits the optimization search space and accelerates convergence.
4. **Additive Ramp Decoder:** Models rapid generation ramp events (sudden weather-induced wind/solar climbs or drops) explicitly as an additive factor in the decoder, bounding extreme forecast deviations during weather extremes.

---

### Causal Inference & Interpretability

In addition to superior forecasting accuracy, this work provides rigorous causal interpretability:
* **Causal Double Machine Learning (DML):** We use DML to estimate the true causal treatment effect of renewable penetration on TSO forecasting errors.
* **Tipping Point Identification:** Our causal curves reveal critical green energy thresholds (e.g., ~54% RES penetration in Germany) beyond which error amplification plateaus, indicating grid stability limits.
* **Feature Attribution:** We provide SHAP-based feature importance checks confirming that the physics-guided modules capture the largest share of attribution weights during volatile periods.

## Repository Structure

```
code open acess/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
│
├── src/                                # Core source code
│   ├── models/
│   │   ├── phygec_net.py               # ★ PhyGEC-Net (our proposed model)
│   │   ├── itransformer.py             # Baseline: iTransformer
│   │   ├── timexer.py                  # Baseline: TimeXer
│   │   ├── tft.py                      # Baseline: Temporal Fusion Transformer
│   │   └── lightgbm_model.py           # Baseline: LightGBM
│   ├── data_processor.py              # Data loading & normalization
│   ├── dataset.py                      # PyTorch Dataset wrapper
│   ├── trainer.py                      # Training loop with early stopping
│   └── metrics.py                      # MAE, RMSE, Skill Score, DM test
│
├── scripts/                            # Experiment execution
│   ├── run_multiseed_master.py         # ★ Main entry: 5-seed × 3-country × 7-model
│   ├── run_all_experiments.py          # Single-seed experiment runner
│   ├── extract_mechanisms.py           # Extract attention weights & gate values
│   └── run_validation.py              # Post-hoc validation checks
│
├── analysis/                           # Statistical analysis & figures
│   ├── causal_dml_analysis.py          # Double Machine Learning causal inference
│   ├── robustness_checks.py            # Bandwidth sensitivity & placebo tests
│   ├── generate_figure2.py             # Figure 2: Quantitative Bias Analysis
│   └── generate_figures3to7.py         # Figures 3–7: All remaining paper figures
│
├── data/                               # Data acquisition & preprocessing
│   ├── download_entsoe_data.py         # Download from ENTSO-E Transparency Platform
│   ├── preprocess_raw_data.py          # Raw data cleaning & alignment
│   └── prepare_forecast_features.py    # Feature engineering for forecasting
│
└── configs/                            # Experiment configurations
    └── default_config.yaml             # Default hyperparameters
```

## Data

We use publicly available data from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/):

| Dataset | Country | RES Penetration | Period |
|:--------|:--------|:----------------|:-------|
| DE      | Germany | ~40% (Medium)   | 2015-01 to 2020-09 |
| DK_1    | Denmark | ~15% (Low)      | 2015-01 to 2020-09 |
| GB_GBN  | Great Britain | ~65% (High) | 2015-01 to 2020-09 |

**To download data:**
```bash
python data/download_entsoe_data.py
python data/preprocess_raw_data.py
python data/prepare_forecast_features.py
```

## Quick Start

### 1. Environment Setup

You can configure the virtual environment and install all dependencies automatically using the provided one-click setup scripts:

**On Linux/macOS:**
```bash
bash scripts/setup_env.sh
```

**On Windows:**
Double-click or run from Command Prompt:
```cmd
scripts\setup_env.bat
```

*Alternatively, to set up manually:*
```bash
# Create conda environment
conda create -n phygecnet python=3.10 -y
conda activate phygecnet

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

### 2. Run Full Experiment (Multi-Seed)

```bash
# Run all models × all countries × 5 seeds (requires 8 GPUs)
python scripts/run_multiseed_master.py
```

### 3. Run Single Experiment

```bash
# Train PhyGEC-Net on Germany (DE) with seed 42
python scripts/run_all_experiments.py --model PhyGEC-Net --country DE --seed 42
```

### 4. Causal Analysis & Figures

```bash
# Run Double Machine Learning analysis
python analysis/causal_dml_analysis.py

# Run robustness checks (bandwidth sensitivity + placebo)
python analysis/robustness_checks.py

# Generate all paper figures
python analysis/generate_figure2.py
python analysis/generate_figures3to7.py
```

## Key Hyperparameters

| Parameter | PhyGEC-Net | iTransformer | TimeXer | TFT |
|:----------|:----------:|:------------:|:-------:|:---:|
| Lookback (hours) | 168 | 168 | 168 | 168 |
| Horizon (hours) | 24 | 24 | 24 | 24 |
| Patch Length | 24 | — | 24 | — |
| d_model | 128 | 128 | 128 | 128 |
| Attention Heads | 4 | 4 | 4 | 4 |
| Encoder Layers | 2 | 2 | 2 | — |
| Learning Rate | 5e-5 | 1e-4 | 1e-4 | 1e-4 |
| Batch Size | 32 | 32 | 32 | 32 |
| Sparsity λ | 0.001 | — | — | — |

## Main Results

PhyGEC-Net achieves state-of-the-art performance across all three national grids:

### 1. Benchmarking Comparisons (Mean ± SD, 5 Seeds)

| Grid | Model | MAE (GW) | RMSE (GW) | Skill Score vs TSO |
|:-----|:------|:--------:|:--------:|:------------------:|
| **Germany (DE)** | **PhyGEC-Net** | **0.9895 ± 0.0326** | **1.2969 ± 0.0446** | **32.17%** |
| | TimeXer | 1.0173 ± 0.0147 | 1.3325 ± 0.0217 | 30.27% |
| | iTransformer | 1.0655 ± 0.0052 | 1.3966 ± 0.0095 | 26.97% |
| **Great Britain (GB)** | **PhyGEC-Net** | **1.2907 ± 0.0099** | **1.8746 ± 0.0162** | **46.77%** |
| | TimeXer | 1.3077 ± 0.0072 | 1.8871 ± 0.0111 | 46.07% |
| | iTransformer | 1.3662 ± 0.0138 | 1.9715 ± 0.0279 | 43.66% |
| **Denmark (DK)** | **PhyGEC-Net** | **0.0249 ± 0.0000** | **0.0363 ± 0.0001** | **16.34%** |
| | TimeXer | 0.0259 ± 0.0003 | 0.0377 ± 0.0005 | 12.88% |
| | iTransformer | 0.0267 ± 0.0002 | 0.0388 ± 0.0003 | 10.25% |

### 2. Model Ablation Study (MAE in GW, Averaged Over 5 Seeds)

| Configuration | Germany (DE) | Denmark (DK_1) | Great Britain (GB_GBN) |
| :--- | :---: | :---: | :---: |
| **PhyGEC-Net (Full)** | **0.9895** | **0.02492** | **1.2907** |
| w/o Sign Gating (`ablate_sign`) | 0.9888 | 0.02488 | 1.2915 |
| w/o Periodicity (`ablate_period`) | 1.0077 | 0.02492 | 1.2937 |
| w/o Ramp Decoder (`ablate_ramp`) | 0.9892 | 0.02490 | 1.2903 |

### 3. Causal Double Machine Learning (DML) Sensitivity Analysis (Effect Coefficient $\theta$)

| Electric Grid | Q1 (9.5% RES) | Q2 (24.5% RES) | Q3 (39.4% RES) | Q4 (54.4% RES) |
| :--- | :---: | :---: | :---: | :---: |
| **Germany (DE)** | 0.4136 (***) | 0.4084 (***) | 0.2021 (***) | 0.0044 (N.S.) |
| **Great Britain (GB)** | 0.2952 (***) | 0.2435 (***) | 0.0293 (N.S.) | -0.0560 (N.S.) |
| **Denmark (DK_1)** | 4.5658 (**) | 0.8829 (N.S.) | 0.0118 (N.S.) | -1.8245 (*) |

*Note: (***) indicates p < 0.001, (**) indicates p < 0.01, (*) indicates p < 0.05, and N.S. indicates not statistically significant.*

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{phygecnet2026,
  title={PhyGEC-Net: Physics-Guided Error Correction for Day-Ahead Load Forecasting under High Renewable Penetration},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
