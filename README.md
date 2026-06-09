# Benchmarking Sensor Robustness in Plasma Diagnostic Models

**Author:** Neerav Gupta

## Overview

This repository contains code and results for the first systematic 
robustness benchmark of plasma diagnostic ML models under realistic 
sensor failure, using the [TokaMark benchmark](https://arxiv.org/abs/2602.10132) 
dataset of 11,573 MAST tokamak shots.

We evaluate three model architectures (XGBoost, LSTM, Transformer) 
across six physically-motivated failure scenarios and three mitigation 
strategies, introducing a Robustness Score (RS) for standardized 
cross-architecture comparison.

## Key Findings

- Disruption-proximate sensor failure causes catastrophic degradation 
  in sequence models (LSTM: +212% NRMSE) while statistical models 
  remain comparatively robust (XGBoost: +37%)
- Forward-fill imputation nearly eliminates degradation from random 
  dropout for sequence models (LSTM: +57% → ~0%)
- Plasma current is the single most critical diagnostic signal 
  (+73% to +140% degradation upon removal)
- Front-loaded acquisition gaps — the dominant natural failure mode 
  in MAST data — cause negligible degradation in all models

## Repository Structure

scripts/
├── config.py              # Central experiment configuration
├── collect_data.py        # Data collection from TokaMark S3
├── corruption.py          # All corruption + mitigation functions
├── feature_engineering.py # Statistical feature extraction (XGBoost)
├── data_loader.py         # Data loading utilities
├── train_xgboost.py       # XGBoost training + evaluation
├── train_lstm.py          # LSTM training + evaluation
├── train_transformer.py   # Transformer training + evaluation
└── analyze_results.py     # Figure generation
results/
├── xgboost_results.json
├── lstm_results.json
└── transformer_results.json
plots/
├── fig1_degradation_curves.png
├── fig2_channel_importance.png
├── fig3_correlated_failure.png
├── fig4_proximate_comparison.png
├── fig5_mitigation_effectiveness.png
└── fig6_robustness_scores.png

## Data & Checkpoints

Full data arrays and trained model checkpoints are available on 
HuggingFace:

**[tokamark-robustness-data](https://huggingface.co/datasets/Neerav-Gupta/tokamark-robustness-data)**

## Reproducing Results

### Requirements

```bash
python -m venv venv --system-site-packages
source venv/bin/activate
pip install torch xgboost scikit-learn zarr s3fs==2024.2.0 \
    botocore==1.34.0 fsspec==2024.2.0 matplotlib seaborn tqdm pandas
cd tokamark && pip install -e .
```

### Run experiments

```bash
# Step 1 — Collect data (streams from UKAEA S3, ~2 hours)
python scripts/collect_data.py

# OR download pre-collected arrays from HuggingFace
python -c "
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(
    repo_id='Neerav-Gupta/tokamark-robustness-data',
    repo_type='dataset',
    local_dir='fusion_research/data'
)
"

# Step 2 — Train models and run experiments
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/train_transformer.py

# Step 3 — Generate figures
python scripts/analyze_results.py
```

## Dependencies

- [TokaMark](https://github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark)
- [FAIR-MAST dataset](https://s3.echo.stfc.ac.uk/mast/tokamark/v1)
- PyTorch, XGBoost, scikit-learn, matplotlib

## Citation

If you use this work, please cite:

```bibtex
@misc{gupta2026tokamark_robustness,
  title={Benchmarking Sensor Robustness in Plasma Diagnostic Models: 
         A Systematic Evaluation on TokaMark},
  author={Gupta, Neerav},
  year={2026},
  note={Preprint}
}
```

Please also cite the TokaMark benchmark:

```bibtex
@article{rousseau2026tokamark,
  title={TokaMark: A Comprehensive Benchmark for MAST Tokamak 
         Plasma Models},
  author={Rousseau, C{\'e}cile and Jackson, Samuel and others},
  journal={arXiv preprint arXiv:2602.10132},
  year={2026}
}
```

## License

MIT License — see LICENSE file.