from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

# ---------------------------------------------------------------------------
# Class name lists — order matches 1-indexed label values in the .npy files.
# Index 0 of each list corresponds to label value 1 (label 0 = discard).
# ---------------------------------------------------------------------------

CLASS_NAMES_7 = [
    "Legume",       # label 1
    "Soy",          # label 2
    "Winter grain", # label 3
    "Maize",        # label 4
    "Potato",       # label 5
    "Squash",       # label 6
    "Other crop",   # label 7
]

CLASS_NAMES_17 = [
    "Winter wheat",      # label 1
    "Winter barley",     # label 2
    "Winter rye",        # label 3
    "Winter triticale",  # label 4
    "Other winter grain",# label 5
    "Spring barley",     # label 6
    "Other spring grain",# label 7
    "Grain maize",       # label 8
    "Silage maize",      # label 9
    "Sunflower",         # label 10
    "Soy",               # label 11
    "Potato",            # label 12
    "Sugar beet",        # label 13
    "Pea",               # label 14
    "Other legume",      # label 15
    "Squash",            # label 16
    "Other crop",        # label 17
]


# ---------------------------------------------------------------------------
# Normalisation constants — computed from the full Austrian dataset.
# Applied per-band across the time dimension for S2, per-channel for S1.
# VIs (NDVI, GCVI, EVI, LSWI) and RVI are NOT normalised (left as-is).
# ---------------------------------------------------------------------------

@dataclass
class NormConstants:
    # Sentinel-2 reflectance bands (10 bands, same order as bands.npy dim-3)
    S2_BAND_MEAN: np.ndarray = field(default_factory=lambda: np.array(
        [1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
         2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922],
        dtype=np.float32,
    ))
    S2_BAND_STD: np.ndarray = field(default_factory=lambda: np.array(
        [1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
         1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779],
        dtype=np.float32,
    ))
    # Sentinel-1 SAR bands: [VV, VH] (RVI is appended by prep_s1_vis.py but not normalised)
    S1_BAND_MEAN: np.ndarray = field(default_factory=lambda: np.array(
        [5484.0407, 3003.7812], dtype=np.float32
    ))
    S1_BAND_STD: np.ndarray = field(default_factory=lambda: np.array(
        [1871.2334, 1726.0670], dtype=np.float32
    ))


# ---------------------------------------------------------------------------
# File paths — all paths are absolute HPC paths passed at runtime via CLI.
# ---------------------------------------------------------------------------

@dataclass
class Paths:
    # Required inputs
    label_file: str              # fieldtype_7classes_maddysmallfields.npy (H, W)
    field_id_file: str           # fieldid.npy (H, W) — one int ID per pixel
    field_data_csv: str          # fielddata_smallfieldlabelgroups.csv
    output_dir: str              # directory for logs, CSVs, and plots

    # Feature-source inputs — only the relevant ones need to be non-empty
    btfm_representations_file: str = ""   # representations_*.npy (H, W, C)
    s2_bands_file: str = ""               # bands.npy or bands_VIs.npy (T, H, W, B)
    sar_asc_file: str = ""                # sar_ascending_VIs.npy (T, H, W, 3)
    sar_desc_file: str = ""               # sar_descending_VIs.npy (T, H, W, 3)
    cloud_mask_file: str = ""             # masks.npy (T, H, W) — binary, 1=valid

    # Optional filtering
    max_dist_csv: str = ""  # CSV with fid_1, max_dist columns for distance filtering


# ---------------------------------------------------------------------------
# Experiment configuration — single source of truth for all run parameters.
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # --- Core settings ---
    feature_source: str = "btfm"   # btfm | raw_s2 | raw_s2s1 | raw_s2s1vis
    model: str = "MLP"             # RandomForest | LogisticRegression | SVM | XGBoost | MLP
    mode: str = "single"           # single | multirun | fewshot
    class_scheme: str = "7class"   # 7class | 17class
    num_repeats: int = 20          # number of seeds in multirun mode (seeds 0..N-1)

    # --- Data split ratios (match source scripts) ---
    training_ratio: float = 0.1        # fraction of per-class area used for training
    val_test_split_ratio: float = 1/7  # fraction of remaining fields used for validation

    # --- Computational settings ---
    chunk_size: int = 1000   # spatial chunk side length in pixels (memory management)
    njobs: int = 12          # parallelism for joblib.Parallel calls

    # --- Optional max-distance filter ---
    max_dist_filter: float = None   # threshold value (None = disabled)
    max_dist_operator: str = "lte"  # comparison direction: lte | gte | lt | gt

    # --- Few-shot settings ---
    num_fields_per_class: int = 50  # N fields per class in fewshot mode

    # --- MLP hyperparameters ---
    # Defaults match austria_classification_toy2_BTFM_multirun.py.
    # For austria_BTFM_mlp_smallfields.py use: [2048,1024,512], lr=0.0005,
    # num_epochs=200, patience=10, use_class_weights=False.
    hidden_sizes: list = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_epochs: int = 50
    patience: int = 5           # early-stopping patience (epochs without val-loss improvement)
    use_class_weights: bool = True  # if True, use sklearn balanced weights in CrossEntropyLoss

    # --- RandomForest hyperparameters ---
    # Defaults match multirun toy scripts (n_estimators=100, unlimited depth).
    # For austria_s1s2_rf_smallfields.py use:
    #   n_estimators=200, max_depth=50, min_samples_split=5, max_features=None
    rf_n_estimators: int = 100
    rf_max_depth: object = None         # int or None (None = grow until leaves are pure)
    rf_min_samples_split: int = 2
    rf_max_features: object = "sqrt"    # 'sqrt' = sklearn default; None = all features

    # --- XGBoost hyperparameters ---
    # Defaults match multirun toy scripts.
    # For austria_BTFM_xgboost_smallfields.py use n_estimators=400.
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # --- SVM hyperparameter tuning (used by tune_hyperparams.py only) ---
    svm_param_grid: object = None   # dict passed to GridSearchCV, or None for fixed SVM
    svm_cv_folds: int = 10          # cross-validation folds for SVM GridSearchCV
