"""Hyperparameter tuning via Cartesian-product search.

Covers the following source scripts:
  austria_classification_toy2_BTFM_tune.py
  austria_classification_toy2_BTFM_tune2.py
  austria_classification_toy2_BTFM_tune_XGBOOST_multi.py
  austria_classification_toy2_BTFM_tune_multi.py
  austria_classification_toy2_raw_tune_RF_multi.py
  austria_classification_toy2_raw_tune_XGBOOST_multi.py

For MLP, RF, XGBoost: iterates over all combinations of the provided
hyperparameter options (seed=0) and saves one CSV row per combination.
For SVM: passes the svm_param_grid to GridSearchCV (single run).

Usage examples
--------------
# MLP tuning
python scripts/tune_hyperparams.py \\
  --feature-source btfm --model MLP \\
  --label-file /path/labels.npy --field-id-file /path/fieldid.npy \\
  --field-data-csv /path/fielddata.csv --output-dir /path/results/ \\
  --btfm-representations-file /path/representations.npy \\
  --hidden-size-options "[512,256,128]" "[1024,512,256]" \\
  --dropout-options 0.3 0.4 0.5 \\
  --lr-options 0.001 0.0005 \\
  --batch-size-options 512 1024

# RF tuning
python scripts/tune_hyperparams.py \\
  --feature-source raw_s2s1vis --model RandomForest \\
  --label-file /path/labels.npy --field-id-file /path/fieldid.npy \\
  --field-data-csv /path/fielddata.csv --output-dir /path/results/ \\
  --s2-bands-file /path/bands_VIs.npy \\
  --rf-estimator-options 100 200 300

# XGBoost tuning
python scripts/tune_hyperparams.py \\
  --feature-source btfm --model XGBoost \\
  --label-file /path/labels.npy --field-id-file /path/fieldid.npy \\
  --field-data-csv /path/fielddata.csv --output-dir /path/results/ \\
  --btfm-representations-file /path/representations.npy \\
  --xgb-estimator-options 100 200 400 \\
  --xgb-depth-options 4 6 8

# SVM tuning with GridSearchCV
python scripts/tune_hyperparams.py \\
  --feature-source btfm --model SVM \\
  --label-file /path/labels.npy --field-id-file /path/fieldid.npy \\
  --field-data-csv /path/fielddata.csv --output-dir /path/results/ \\
  --btfm-representations-file /path/representations.npy \\
  --svm-param-grid '{"C":[0.1,1,10],"gamma":["scale","auto"]}' \\
  --cv-folds 10
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add project root so `smallfields` and sibling scripts are importable
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_scripts_dir)
sys.path.insert(0, _root_dir)
sys.path.insert(0, _scripts_dir)  # allows `from run_experiment import ...`

from smallfields.config import (
    CLASS_NAMES_7,
    CLASS_NAMES_17,
    ExperimentConfig,
    NormConstants,
    Paths,
)
# Reuse run_single and setup_logging from run_experiment to avoid code duplication
from run_experiment import run_single, setup_logging


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter tuning via Cartesian-product grid search."
    )

    # --- Shared args (same as run_experiment.py) ---
    p.add_argument("--feature-source", required=True,
                   choices=["btfm", "raw_s2", "raw_s2s1", "raw_s2s1vis"])
    p.add_argument("--model", required=True,
                   choices=["RandomForest", "LogisticRegression", "SVM", "XGBoost", "MLP"])
    p.add_argument("--class-scheme", default="7class", choices=["7class", "17class"])

    # Data paths
    p.add_argument("--label-file", required=True)
    p.add_argument("--field-id-file", required=True)
    p.add_argument("--field-data-csv", required=True)
    p.add_argument("--output-dir", required=True)

    # Feature-source-specific paths
    p.add_argument("--btfm-representations-file", default="")
    p.add_argument("--s2-bands-file", default="")
    p.add_argument("--sar-asc-file", default="")
    p.add_argument("--sar-desc-file", default="")
    p.add_argument("--cloud-mask-file", default="")

    # Optional max-dist filter
    p.add_argument("--max-dist-filter", type=float, default=None)
    p.add_argument("--max-dist-operator", default="lte",
                   choices=["lte", "gte", "lt", "gt"])
    p.add_argument("--max-dist-csv", default="")

    # General run settings (fixed across all tuning combinations)
    p.add_argument("--training-ratio", type=float, default=0.1)
    p.add_argument("--val-test-split-ratio", type=float, default=1 / 7)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--njobs", type=int, default=12)

    # --- MLP hyperparameter options (nargs='+') ---
    # Each --hidden-size-options value is a JSON list string, e.g. "[512,256,128]"
    p.add_argument(
        "--hidden-size-options", nargs="+", default=None,
        metavar="JSON_LIST",
        help='Space-separated JSON lists, e.g. "[512,256,128]" "[1024,512,256]".',
    )
    p.add_argument("--dropout-options", nargs="+", type=float, default=None)
    p.add_argument("--lr-options", nargs="+", type=float, default=None)
    p.add_argument("--batch-size-options", nargs="+", type=int, default=None)
    # Fixed MLP defaults used for dimensions not being swept
    p.add_argument("--hidden-sizes", type=int, nargs=3, default=[512, 256, 128])
    p.add_argument("--dropout-rate", type=float, default=0.3)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)

    # --- RF hyperparameter options ---
    # Only n_estimators is swept; depth/min_samples_split can be set as fixed defaults
    p.add_argument("--rf-estimator-options", nargs="+", type=int, default=None)
    p.add_argument("--rf-n-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-min-samples-split", type=int, default=2)
    p.add_argument("--rf-max-features", default="sqrt",
                   help="max_features for RF. Use 'none' for all features.")

    # --- XGBoost hyperparameter options ---
    p.add_argument("--xgb-estimator-options", nargs="+", type=int, default=None)
    p.add_argument("--xgb-depth-options", nargs="+", type=int, default=None)
    p.add_argument("--xgb-n-estimators", type=int, default=100)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.1)

    # --- SVM GridSearchCV ---
    # SVM does not use a manual Cartesian product; instead it passes the grid
    # directly to GridSearchCV (as in the source tune scripts).
    p.add_argument(
        "--svm-param-grid", default=None,
        help='JSON dict for SVM GridSearchCV, e.g. \'{"C":[0.1,1,10],"gamma":["scale"]}\'.',
    )
    p.add_argument("--cv-folds", type=int, default=10)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Build hyperparameter combinations
# ---------------------------------------------------------------------------

def build_mlp_combinations(args):
    """Return list of dicts, one per MLP hyperparameter combination.

    Uses itertools.product to enumerate the full Cartesian product across all
    provided option lists.  If an option list is absent, the fixed default is
    used (so the sweep collapses to a single value for that dimension).
    """
    # Parse JSON list strings into Python lists; fall back to the fixed default
    hidden_options = (
        [json.loads(s) for s in args.hidden_size_options]
        if args.hidden_size_options else [args.hidden_sizes]
    )
    dropout_options   = args.dropout_options   or [args.dropout_rate]
    lr_options        = args.lr_options        or [args.learning_rate]
    batchsz_options   = args.batch_size_options or [args.batch_size]

    combos = []
    for hidden, dropout, lr, bs in itertools.product(
        hidden_options, dropout_options, lr_options, batchsz_options
    ):
        combos.append({
            "hidden_sizes": hidden,
            "dropout_rate": dropout,
            "learning_rate": lr,
            "batch_size": bs,
        })
    return combos


def build_rf_combinations(args):
    """Return list of dicts, one per RF hyperparameter combination."""
    n_est_options = args.rf_estimator_options or [args.rf_n_estimators]
    combos = []
    for n_est in n_est_options:
        combos.append({"rf_n_estimators": n_est})
    return combos


def build_xgb_combinations(args):
    """Return list of dicts, one per XGBoost hyperparameter combination."""
    n_est_options  = args.xgb_estimator_options or [args.xgb_n_estimators]
    depth_options  = args.xgb_depth_options     or [args.xgb_max_depth]
    combos = []
    for n_est, depth in itertools.product(n_est_options, depth_options):
        combos.append({"xgb_n_estimators": n_est, "xgb_max_depth": depth})
    return combos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log_file = setup_logging(args.output_dir, args.feature_source, args.model, "tune")
    logging.info(f"Started hyperparameter tuning. Log: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    log_basename = os.path.basename(log_file)
    plot_prefix = os.path.splitext(log_basename)[0]

    paths = Paths(
        label_file=args.label_file,
        field_id_file=args.field_id_file,
        field_data_csv=args.field_data_csv,
        output_dir=args.output_dir,
        btfm_representations_file=args.btfm_representations_file,
        s2_bands_file=args.s2_bands_file,
        sar_asc_file=args.sar_asc_file,
        sar_desc_file=args.sar_desc_file,
        cloud_mask_file=args.cloud_mask_file,
        max_dist_csv=args.max_dist_csv,
    )

    norm = NormConstants()
    class_names = CLASS_NAMES_7 if args.class_scheme == "7class" else CLASS_NAMES_17

    # --- Build base config fields shared across all combinations ---
    # Each combo dict will override only its specific hyperparameters.
    base_cfg_kwargs = dict(
        feature_source=args.feature_source,
        model=args.model,
        mode="single",       # tuning always uses seed=0, single run
        class_scheme=args.class_scheme,
        training_ratio=args.training_ratio,
        val_test_split_ratio=args.val_test_split_ratio,
        chunk_size=args.chunk_size,
        njobs=args.njobs,
        max_dist_filter=args.max_dist_filter,
        max_dist_operator=args.max_dist_operator,
        # MLP fixed defaults (overridden per-combination for the swept dimensions)
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        # RF fixed defaults
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_split=args.rf_min_samples_split,
        # Convert 'none'/'null' string to Python None for sklearn
        rf_max_features=None if args.rf_max_features.lower() in ("none", "null") else args.rf_max_features,
        # XGBoost fixed defaults
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
    )

    model_lower = args.model.lower()

    # --- Build per-model combinations ---
    if model_lower == "mlp":
        combos = build_mlp_combinations(args)
    elif model_lower == "randomforest":
        combos = build_rf_combinations(args)
    elif model_lower == "xgboost":
        combos = build_xgb_combinations(args)
    elif model_lower == "svm":
        # SVM tuning delegates to GridSearchCV inside fit_classifier;
        # we only create one "combination" that carries the full param grid.
        svm_grid = json.loads(args.svm_param_grid) if args.svm_param_grid else None
        combos = [{"svm_param_grid": svm_grid, "svm_cv_folds": args.cv_folds}]
    else:
        # LogisticRegression — no user-facing hyperparams to tune; single run
        combos = [{}]

    logging.info(f"Total hyperparameter combinations: {len(combos)}")

    # --- Run each combination ---
    all_rows = []
    for idx, combo in enumerate(combos):
        logging.info(f"--- Combination {idx + 1}/{len(combos)}: {combo} ---")

        # Merge base config with this combination's overrides
        cfg_kwargs = dict(base_cfg_kwargs)
        cfg_kwargs.update(combo)
        cfg = ExperimentConfig(**cfg_kwargs)

        # All tuning runs use seed=0 and skip plot generation for speed
        metrics = run_single(
            seed=0,
            paths=paths,
            cfg=cfg,
            norm=norm,
            class_names=class_names,
            device=device,
            save_plots=False,
        )

        # Build a flat dict for this CSV row: hyperparams + metrics
        row = dict(combo)
        # Serialise list/dict values so they display cleanly in the CSV
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                row[k] = json.dumps(v)
        row.update(metrics)
        all_rows.append(row)
        logging.info(f"Combination {idx + 1} result: {metrics}")

    # --- Save results ---
    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        args.output_dir,
        f"{args.feature_source}_{model_lower}_tune_results_{ts}.csv",
    )
    df.to_csv(out_csv, index=False)
    logging.info(f"Tuning results saved to {out_csv}")
    print(df.to_string())
    logging.info("Done.")


if __name__ == "__main__":
    main()
