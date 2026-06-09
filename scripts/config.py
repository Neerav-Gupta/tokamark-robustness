"""
Central configuration for all TokaMark robustness benchmark experiments.
Update paths if running outside the default RunPod workspace.
"""

# TokaMark S3 data source (UKAEA public endpoint)
STORE_SETTINGS = {
    "s3_endpoint_url": "https://s3.echo.stfc.ac.uk",
    "s3_mast_dataset_path": "/mast/tokamark/v1",
    "base_fsspec_protocol": "simplecache",
    "target_fsspec_protocol": "s3",
}

TASK_NAME = "task_4-4"  # Plasma current quench prediction

# Corruption experiment grid
DROP_RATES = [0.10, 0.25, 0.50]
GAP_FRACTIONS = [0.20, 0.40, 0.60]
N_CHANNELS_TO_KILL = [1, 3, 6]
CORRELATED_GROUPS = ["kinetics", "magnetics_active", "radiatives", "mirnov"]

RANDOM_SEED = 42

# Output paths
RESULTS_DIR = "/workspace/fusion_research/results"
CHECKPOINTS_DIR = "/workspace/fusion_research/checkpoints"
PLOTS_DIR = "/workspace/fusion_research/plots"

# TokaMark package source
TOKAMARK_SRC = "/workspace/tokamark/src"