"""Main CLI entry point for running crop-type classification experiments.

Usage
-----
python scripts/run_experiment.py \\
  --feature-source btfm \\
  --model MLP \\
  --mode multirun \\
  --class-scheme 7class \\
  --num-repeats 20 \\
  --label-file /path/to/labels.npy \\
  --field-id-file /path/to/fieldid.npy \\
  --field-data-csv /path/to/fielddata.csv \\
  --output-dir /path/to/results/ \\
  --btfm-representations-file /path/to/representations.npy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from sklearn.metrics import classification_report, f1_score

# Add project root to path so `smallfields` is importable without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smallfields.config import (
    CLASS_NAMES_7,
    CLASS_NAMES_17,
    ExperimentConfig,
    NormConstants,
    Paths,
)
from smallfields.data.features import FeatureLoader
from smallfields.data.loading import (
    apply_max_dist_filter,
    identify_valid_classes,
    load_field_data,
    load_labels_and_fields,
)
from smallfields.data.splitting import (
    area_weighted_split,
    create_split_mask,
    fewshot_split,
)
from smallfields.evaluation.metrics import (
    collect_run_metrics,
    compute_mean_ci,
    compute_per_class_accuracy,
    compute_pixel_accuracy,
)
from smallfields.evaluation.visualization import (
    get_color_palette,
    plot_classification_map,
    plot_confusion_matrix,
    plot_per_class_accuracy,
)
from smallfields.models.classifiers import fit_classifier
from smallfields.pipeline.chunk_processing import (
    combine_chunk_results,
    generate_chunks,
    process_chunk,
)
from smallfields.pipeline.prediction import batch_predict_chunk, build_prediction_map


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Run a crop classification experiment.")

    # Feature source
    p.add_argument("--feature-source", required=True,
                   choices=["btfm", "raw_s2", "raw_s2s1", "raw_s2s1vis"],
                   help="Which feature representation to use.")
    p.add_argument("--model", required=True,
                   choices=["RandomForest", "LogisticRegression", "SVM", "XGBoost", "MLP"],
                   help="Classifier to train.")
    p.add_argument("--mode", default="single",
                   choices=["single", "multirun", "fewshot"],
                   help="Experiment mode.")
    p.add_argument("--class-scheme", default="7class",
                   choices=["7class", "17class"])
    p.add_argument("--num-repeats", type=int, default=20,
                   help="Number of repeats for multirun mode.")

    # Data paths
    p.add_argument("--label-file", required=True)
    p.add_argument("--field-id-file", required=True)
    p.add_argument("--field-data-csv", required=True)
    p.add_argument("--output-dir", required=True)

    # Feature-source-specific paths (unused paths are silently ignored by loaders)
    p.add_argument("--btfm-representations-file", default="")
    p.add_argument("--s2-bands-file", default="")
    p.add_argument("--sar-asc-file", default="")
    p.add_argument("--sar-desc-file", default="")
    p.add_argument("--cloud-mask-file", default="")

    # Optional max-dist filter — restricts training/eval to fields with max_dist <= threshold
    p.add_argument("--max-dist-filter", type=float, default=None,
                   help="Max distance threshold for field filtering.")
    p.add_argument("--max-dist-operator", default="lte",
                   choices=["lte", "gte", "lt", "gt"])
    p.add_argument("--max-dist-csv", default="")

    # Fewshot
    p.add_argument("--num-fields-per-class", type=int, default=50,
                   help="Fields per class in fewshot mode.")

    # General hyperparameters
    p.add_argument("--training-ratio", type=float, default=0.1)
    # 1/7 of remaining (val+test) fields go to validation; matches source default
    p.add_argument("--val-test-split-ratio", type=float, default=1/7)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--njobs", type=int, default=12)

    # MLP hyperparameters
    # Default [512,256,128] matches austria_classification_toy2_BTFM_multirun.py.
    # For austria_BTFM_mlp_smallfields.py use [2048,1024,512] with lr=0.0005, epochs=200.
    p.add_argument("--hidden-sizes", type=int, nargs=3, default=[512, 256, 128])
    p.add_argument("--dropout-rate", type=float, default=0.3)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)

    # RandomForest hyperparameters
    # Defaults match multirun toy scripts. For austria_s1s2_rf_smallfields.py use
    # n_estimators=200, max_depth=50, min_samples_split=5, max_features=none.
    p.add_argument("--rf-n-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=None,
                   help="Max depth for RandomForest (omit for unlimited).")
    p.add_argument("--rf-min-samples-split", type=int, default=2)
    p.add_argument("--rf-max-features", default="sqrt",
                   help="max_features for RandomForest. Use 'sqrt' (default), "
                        "'none' (all features, matches austria_s1s2_rf_smallfields.py), "
                        "or a float/int.")

    # XGBoost hyperparameters
    # Default n_estimators=100 matches multirun toys; austria_BTFM_xgboost_smallfields.py uses 400.
    p.add_argument("--xgb-n-estimators", type=int, default=100)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.1)

    # MLP class-weight flag
    # Default True matches multirun scripts; use --no-use-class-weights for
    # austria_BTFM_mlp_smallfields.py (which comments out class weights)
    p.add_argument("--use-class-weights", dest="use_class_weights",
                   action="store_true", default=True,
                   help="Use balanced class weights in MLP CrossEntropyLoss (default: True).")
    p.add_argument("--no-use-class-weights", dest="use_class_weights",
                   action="store_false",
                   help="Disable class weights (matches austria_BTFM_mlp_smallfields.py).")

    # Visualization — enabled by default; disable with --no-save-plots for speed
    p.add_argument("--save-plots", action="store_true", default=True,
                   help="Save confusion matrix, per-class accuracy, and classification map PNGs.")
    p.add_argument("--no-save-plots", dest="save_plots", action="store_false")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _parse_rf_max_features(value: str):
    """Convert CLI string to the value expected by sklearn RandomForestClassifier.

    'none' or 'null' → None (all features, matches austria_s1s2_rf_smallfields.py)
    'sqrt', 'log2'   → string (sklearn keywords)
    numeric string   → float (if contains '.') or int
    """
    if value.lower() in ("none", "null"):
        return None
    # Try int first (e.g. '10' → use exactly 10 features)
    try:
        return int(value)
    except ValueError:
        pass
    # Try float next (e.g. '0.5' → use 50% of features)
    try:
        return float(value)
    except ValueError:
        pass
    return value  # e.g. 'sqrt', 'log2' — passed through to sklearn as-is


def setup_logging(output_dir: str, feature_source: str, model: str, mode: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    # Timestamp in filename ensures each run's log is uniquely named
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"{feature_source}_{model}_{mode}_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # write mode: new file each run
            logging.StreamHandler(),                   # also echo to stdout
        ],
    )
    return log_file


# ---------------------------------------------------------------------------
# Core single-run logic
# ---------------------------------------------------------------------------

def run_single(
    seed: int,
    paths: Paths,
    cfg: ExperimentConfig,
    norm: NormConstants,
    class_names: list,
    device,
    save_plots: bool = False,
    plot_prefix: str = "",
    is_single_run: bool = False,
) -> dict:
    """Execute one full train-evaluate cycle and return metrics dict."""
    # Set all random seeds for full reproducibility across numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # XGBoost requires 0-indexed labels (0 to num_classes-1).
    # All other models use 1-indexed labels (1 to num_classes); 0 = background.
    zero_index = cfg.model.lower() == "xgboost"
    labels, field_ids, H, W = load_labels_and_fields(
        paths.label_file, paths.field_id_file, zero_index_labels=zero_index
    )
    # With zero-indexing, the old background (0) becomes -1 after the shift
    background_value = -1 if zero_index else 0

    # --- Optional max-dist filter ---
    # Zeros out pixels of excluded fields so they are treated as background
    filtered_fids_set = None
    if cfg.max_dist_filter is not None and paths.max_dist_csv:
        field_ids, filtered_fids_set = apply_max_dist_filter(
            field_ids, paths.max_dist_csv, cfg.max_dist_filter, cfg.max_dist_operator
        )

    # Classes with fewer than 2 pixels are excluded to prevent splits with 0 samples
    valid_classes = identify_valid_classes(labels, background_value=background_value)

    # 7-class scheme uses remapped_SNAR_CODE; 17-class uses the original SNAR_CODE
    snar_col = "remapped_SNAR_CODE" if cfg.class_scheme == "7class" else "SNAR_CODE"
    fielddata_df = load_field_data(paths.field_data_csv, filtered_fids=filtered_fids_set)

    # --- Split ---
    if cfg.mode == "fewshot":
        train_fids, val_fids, test_fids = fewshot_split(
            fielddata_df,
            snar_code_column=snar_col,
            fid_column="fid_1",
            num_fields_per_class=cfg.num_fields_per_class,
            seed=seed,
        )
    else:
        # area_weighted_split is used for both 'single' and 'multirun' modes
        train_fids, val_fids, test_fids = area_weighted_split(
            fielddata_df,
            snar_code_column=snar_col,
            fid_column="fid_1",
            area_column="area_m2",
            training_ratio=cfg.training_ratio,
            val_test_split_ratio=cfg.val_test_split_ratio,
            seed=seed,
        )

    logging.info(
        f"Seed {seed} — train: {len(train_fids)}, val: {len(val_fids)}, test: {len(test_fids)} fields"
    )

    # --- Feature loader ---
    # All file paths are bundled into kwargs so FeatureLoader is fully self-contained
    # and picklable for joblib.Parallel dispatch to subprocesses.
    loader_kwargs = dict(
        representations_file=paths.btfm_representations_file,
        bands_file=paths.s2_bands_file,
        sar_asc_file=paths.sar_asc_file,
        sar_desc_file=paths.sar_desc_file,
        cloud_mask_file=paths.cloud_mask_file,
        norm=norm,
    )
    feature_loader = FeatureLoader(cfg.feature_source, **loader_kwargs)

    # --- Chunk processing ---
    # The full (H, W) image is processed in spatial tiles to limit peak memory usage.
    # Each chunk is dispatched to a worker process via joblib.Parallel.
    chunks = generate_chunks(H, W, cfg.chunk_size)
    results = Parallel(n_jobs=cfg.njobs)(
        delayed(process_chunk)(
            h_start, h_end, w_start, w_end,
            labels, field_ids, valid_classes,
            train_fids, val_fids, test_fids,
            feature_loader,
        )
        for h_start, h_end, w_start, w_end in chunks
    )
    X_train, y_train, X_val, y_val, X_test, y_test = combine_chunk_results(results)
    logging.info(
        f"Pixels — train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    # --- Model training ---
    mlp_kwargs = dict(
        hidden_sizes=cfg.hidden_sizes,
        dropout_rate=cfg.dropout_rate,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        num_epochs=cfg.num_epochs,
        patience=cfg.patience,
        use_class_weights=cfg.use_class_weights,
    )
    model = fit_classifier(
        cfg.model,
        X_train, y_train, X_val, y_val,
        valid_classes=valid_classes,
        njobs=cfg.njobs,
        mlp_kwargs=mlp_kwargs,
        device=device,
        rf_n_estimators=cfg.rf_n_estimators,
        rf_max_depth=cfg.rf_max_depth,
        rf_min_samples_split=cfg.rf_min_samples_split,
        rf_max_features=cfg.rf_max_features,
        xgb_n_estimators=cfg.xgb_n_estimators,
        xgb_max_depth=cfg.xgb_max_depth,
        xgb_learning_rate=cfg.xgb_learning_rate,
        svm_param_grid=cfg.svm_param_grid,
        svm_cv_folds=cfg.svm_cv_folds,
    )

    # --- Evaluation on test set ---
    y_pred = model.predict(X_test)
    report_str = classification_report(y_test, y_pred, digits=4)
    logging.info("Classification Report (Test Set):\n" + report_str)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)
    # weighted avg F1 is stored as 'balanced_f1' in the output CSV (source convention)
    f1_weighted = report["weighted avg"]["f1-score"]
    f1_macro = report["macro avg"]["f1-score"]

    # --- Pixel accuracy via prediction map ---
    # Build the full spatial prediction map, then compute accuracy from it.
    # This matches the source scripts which also compute accuracy from the raster map
    # (rather than directly from y_test/y_pred arrays).
    pred_results = Parallel(n_jobs=cfg.njobs)(
        delayed(batch_predict_chunk)(
            h_start, h_end, w_start, w_end,
            labels, field_ids, valid_classes, train_fids, model, feature_loader,
        )
        for h_start, h_end, w_start, w_end in chunks
    )
    pred_map = build_prediction_map(pred_results, H, W)

    test_accuracy = compute_pixel_accuracy(labels, pred_map, field_ids, valid_classes, test_fids)
    val_accuracy = compute_pixel_accuracy(labels, pred_map, field_ids, valid_classes, val_fids)

    logging.info(
        f"Test accuracy: {test_accuracy:.2f}%  Val accuracy: {val_accuracy:.2f}%"
    )

    # --- Visualization ---
    if save_plots:
        # Build a colormap aligned with the class label integers (max class = colormap length)
        max_cls = max(valid_classes) if valid_classes else 1
        colors = get_color_palette(max_cls + 1)
        if not zero_index:
            colors[0] = (1, 1, 1, 1)  # white for background (label 0)
        cmap = ListedColormap(colors)

        # Confusion matrix: computed from pixel-level y_test/y_pred arrays
        cm_path = os.path.join(
            paths.output_dir, f"{plot_prefix}_seed{seed}_confusion_matrix.png"
        )
        plot_confusion_matrix(
            y_test, y_pred, valid_classes, class_names,
            f"{cfg.model} Confusion Matrix (seed={seed})", cm_path,
        )
        logging.info(f"Confusion matrix saved to {cm_path}")

        # Per-class accuracy: computed from the full spatial prediction map
        class_accs = compute_per_class_accuracy(
            labels, pred_map, field_ids, valid_classes, test_fids
        )
        pca_path = os.path.join(
            paths.output_dir, f"{plot_prefix}_seed{seed}_per_class_accuracy.png"
        )
        plot_per_class_accuracy(
            class_accs, class_names, test_accuracy, cfg.model, cmap, pca_path
        )
        logging.info(f"Per-class accuracy chart saved to {pca_path}")

        # Classification map and raw pred_map .npy are only saved in single-run mode
        # (in multirun, the last seed's map would overwrite previous ones)
        if is_single_run:
            map_path = os.path.join(
                paths.output_dir, f"{plot_prefix}_classification_map.png"
            )
            plot_classification_map(
                pred_map, f"{cfg.model} Classification Map", cmap, class_names, map_path
            )
            logging.info(f"Classification map saved to {map_path}")

            # Save raw prediction array for downstream analysis
            pred_npy_path = os.path.join(
                paths.output_dir, f"{plot_prefix}_pred_map.npy"
            )
            np.save(pred_npy_path, pred_map)
            logging.info(f"Prediction map saved to {pred_npy_path}")

    return collect_run_metrics(seed, val_accuracy, test_accuracy, f1_weighted, f1_macro)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log_file = setup_logging(args.output_dir, args.feature_source, args.model, args.mode)
    logging.info(f"Started. Log: {log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Derive timestamp prefix from log filename for consistent plot/CSV naming
    log_basename = os.path.basename(log_file)
    plot_prefix = os.path.splitext(log_basename)[0]  # strip .log extension

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

    cfg = ExperimentConfig(
        feature_source=args.feature_source,
        model=args.model,
        mode=args.mode,
        class_scheme=args.class_scheme,
        num_repeats=args.num_repeats,
        training_ratio=args.training_ratio,
        val_test_split_ratio=args.val_test_split_ratio,
        chunk_size=args.chunk_size,
        njobs=args.njobs,
        max_dist_filter=args.max_dist_filter,
        max_dist_operator=args.max_dist_operator,
        num_fields_per_class=args.num_fields_per_class,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        patience=args.patience,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        rf_min_samples_split=args.rf_min_samples_split,
        # Convert CLI string (e.g. 'none', 'sqrt', '0.5') to the value sklearn expects
        rf_max_features=_parse_rf_max_features(args.rf_max_features),
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_learning_rate=args.xgb_learning_rate,
        use_class_weights=args.use_class_weights,
    )

    norm = NormConstants()
    class_names = CLASS_NAMES_7 if args.class_scheme == "7class" else CLASS_NAMES_17

    # --- Determine seeds to iterate over ---
    # single and fewshot modes run exactly one seed (0); multirun iterates 0..N-1
    if args.mode == "single" or args.mode == "fewshot":
        seeds = [0]
    else:  # multirun
        seeds = list(range(args.num_repeats))

    # Classification map is only saved in single-seed runs (one definitive output)
    is_single_run = len(seeds) == 1

    # --- Run ---
    output_results = []
    for seed in seeds:
        logging.info(f"--- Run seed={seed} ---")
        metrics = run_single(
            seed, paths, cfg, norm, class_names, device,
            save_plots=args.save_plots,
            plot_prefix=plot_prefix,
            is_single_run=is_single_run,
        )
        output_results.append(metrics)

    # --- Save per-run CSV ---
    # One row per seed; columns: random_seed, validation_accuracy, test_accuracy,
    # balanced_f1, macro_f1.  Matches the format consumed by summarize_results.py.
    df_results = pd.DataFrame(output_results)
    run_csv = os.path.join(
        args.output_dir,
        f"{args.feature_source}_{args.model.lower()}_run_results.csv",
    )
    df_results.to_csv(run_csv, index=False)
    logging.info(f"Per-run results saved to {run_csv}")

    # --- Summary statistics (multirun only) ---
    if args.mode == "multirun" and len(output_results) > 1:
        summary_rows = []
        for metric in ["validation_accuracy", "test_accuracy", "balanced_f1", "macro_f1"]:
            mean, ci = compute_mean_ci(df_results[metric])
            logging.info(f"{metric}: {mean:.4f} ± {ci:.4f} (95% CI)")
            summary_rows.append({
                "model": args.model.lower(),
                "metric": metric,
                "mean": round(mean, 4),
                "95%_CI": round(ci, 4),
            })

        df_summary = pd.DataFrame(summary_rows)
        # *allmodels_summary_stats.csv naming matches the glob in summarize_results.py
        summary_csv = os.path.join(
            args.output_dir,
            f"{args.feature_source}_allmodels_summary_stats.csv",
        )
        df_summary.to_csv(summary_csv, index=False)
        logging.info(f"Summary stats saved to {summary_csv}")
        print(df_summary)

    logging.info("Done.")


if __name__ == "__main__":
    main()
