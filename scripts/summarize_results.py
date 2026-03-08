"""Aggregate per-treatment multirun CSVs into a multi-index summary table.

Usage
-----
python scripts/summarize_results.py \\
  --input-dir /path/to/results/ \\
  --output-csv /path/to/results/summary_metrics_multiindex.csv

The script reads all files matching ``*allmodels_summary_stats.csv`` in
``--input-dir`` and maps each file's prefix to a treatment name using the
same convention as the source ``create_summary_table.py``:

  btfm*        → "Representations"
  rawcmvis*    → "Raw + cloudmask and VIs"
  rawcm*       → "Raw + cloudmask"
  raw*         → "Raw"
  (other)      → filename stem

It then pivots mean / 95%_CI per model per metric and writes a multi-index
CSV (treatment × metric at columns, model at index).
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


# Order matters: more-specific prefixes must come before their substrings.
# e.g. 'rawcmvis' must precede 'rawcm' and 'raw' so that a filename like
# 'rawcmvis_allmodels_summary_stats.csv' is matched by the correct rule first.
TREATMENT_MAP = [
    ("rawcmvis", "Raw + cloudmask and VIs"),
    ("rawcm",    "Raw + cloudmask"),
    ("raw",      "Raw"),
    ("btfm",     "Representations"),
]

# Only these three metrics are included in the final summary table
METRICS_OF_INTEREST = {
    "test_accuracy": "Test Accuracy",
    "macro_f1":      "Macro F1",
    "balanced_f1":   "Balanced F1",
}


def filename_to_treatment(stem: str) -> str:
    """Map a CSV filename stem to a human-readable treatment name."""
    for prefix, name in TREATMENT_MAP:
        if stem.startswith(prefix):
            return name
    # No recognised prefix — use the raw filename stem as the treatment name
    return stem


def load_summary_table(path: str) -> pd.DataFrame:
    """Load one *allmodels_summary_stats.csv and pivot it to a model-indexed DataFrame."""
    # Derive treatment label from the filename prefix (e.g. 'btfm' → 'Representations')
    stem = os.path.basename(path).replace(".csv", "")
    treatment_name = filename_to_treatment(stem)

    df = pd.read_csv(path)

    # Drop metrics not in our summary list (e.g. validation_accuracy)
    df = df[df["metric"].isin(METRICS_OF_INTEREST.keys())]

    # Pivot: index=model, columns=(stat × metric), values=mean and 95%_CI
    pivot = df.pivot(index="model", columns="metric", values=["mean", "95%_CI"])

    # Combine mean and CI into a single "mean ± CI" string column per metric
    combined = pd.DataFrame(index=pivot.index)
    columns = []
    for metric_key, pretty_name in METRICS_OF_INTEREST.items():
        mean_col = ("mean", metric_key)
        ci_col = ("95%_CI", metric_key)
        # Format as "42.35 ± 0.87" — matches source create_summary_table.py output
        combined_col = pivot.apply(
            lambda row: f"{row[mean_col]:.2f} ± {row[ci_col]:.2f}", axis=1
        )
        col_tuple = (treatment_name, pretty_name)
        columns.append(col_tuple)
        combined[col_tuple] = combined_col

    # Build a MultiIndex header: (treatment, metric_name)
    combined.columns = pd.MultiIndex.from_tuples(columns)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Aggregate multirun summary CSVs.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing *allmodels_summary_stats.csv files.")
    parser.add_argument("--output-csv", required=True,
                        help="Path for the combined multi-index output CSV.")
    args = parser.parse_args()

    # Collect all per-treatment summary files in the results directory
    pattern = os.path.join(args.input_dir, "*allmodels_summary_stats.csv")
    csv_paths = sorted(glob.glob(pattern))

    if not csv_paths:
        print(f"No files matching '{pattern}' found.")
        return

    # Load each treatment file and pivot to a model-indexed DataFrame
    tables = [load_summary_table(p) for p in csv_paths]
    # Concatenate along columns to produce a wide table: models × (treatment, metric)
    final_summary = pd.concat(tables, axis=1)
    final_summary.index.name = "Model"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    # utf-8-sig BOM ensures Excel opens the CSV with correct encoding on Windows
    final_summary.to_csv(args.output_csv, encoding="utf-8-sig")
    print(f"Summary table saved to {args.output_csv}")

    # Print the full table to stdout with all columns visible
    with pd.option_context("display.max_columns", None):
        print(final_summary)


if __name__ == "__main__":
    main()
