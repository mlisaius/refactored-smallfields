# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **crop type classification of small agricultural fields in Austria** using satellite imagery. It compares two main feature representations:
- **Raw satellite data** (Sentinel-1 SAR + Sentinel-2 optical, with optional vegetation indices)
- **BTFM representations** (Bio-Temporal Foundation Model embeddings from a pre-trained model)

The `source-temp/` directory contains the original source scripts that are to be refactored into a clean, modular codebase in this repository.

## Environment & Running Scripts

Scripts run on a Linux HPC cluster (paths like `/maps/mcl66/`, `/maps/zf281/`, `/home/mcl66/`). There is no package manager config — dependencies are standard Python scientific stack:

```
numpy, pandas, scikit-learn, xgboost, torch, h5py, matplotlib, seaborn, scipy, joblib
```

Scripts are run directly: `python <script_name>.py`

There are no build steps, test suites, or linters configured.

## Data Architecture

All data is stored as `.npy` arrays (field-indexed, not pixel-indexed):

| File | Description |
|------|-------------|
| `bands.npy` / `bands_VIs.npy` | Sentinel-2 optical bands (shape: `T×H×W×B`, 10 bands + optional 4 VIs) |
| `sar_ascending_VIs.npy` / `sar_descending_VIs.npy` | Sentinel-1 SAR data (VV, VH, + RVI) |
| `representations_*.npy` | BTFM embeddings (shape: `H×W×C`) |
| `fieldtype_7classes_maddysmallfields.npy` | Crop type labels (7 classes: 1=Legume, 2=Soy, 3=Winter grain, 4=Maize, 5=Potato, 6=Squash, 7=Other crop; 0=Discard) |
| `fieldid.npy` | Field IDs for grouping pixels per field |
| `fielddata_smallfieldlabelgroups.csv` | Field metadata CSV |

**Key data note**: Labels use 1-indexed classes; class 0 is "discard". When passed to PyTorch, labels are shifted by `-1` to become 0-indexed.

## Classification Approach

All classification scripts follow this pipeline:
1. Load `.npy` feature data + labels + field IDs
2. Filter out discarded pixels (label == 0)
3. Split by **field** (not pixel) to prevent data leakage — `TRAINING_RATIO=0.1`, `VAL_TEST_SPLIT_RATIO=1/7`
4. Normalize features (StandardScaler or fixed S1/S2 band statistics)
5. Train classifier: XGBoost, RandomForest, LogisticRegression, SVM, or MLP
6. Evaluate with accuracy, macro F1, balanced F1, confusion matrix
7. Output logs, CSV results, and confusion matrix plots

**Few-shot variants** (`*_fewshot.py`) test with very limited training data per class.

**Multirun variants** (`*_multirun.py`) repeat training 20 times with different random seeds and report 95% CI.

## Script Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `austria_BTFM_*` | Uses BTFM representation embeddings as features |
| `austria_s1s2_*` | Uses raw Sentinel-1 + Sentinel-2 bands as features |
| `austria_classification_toy2_BTFM_*` | BTFM features, downsampled/toy dataset |
| `austria_classification_toy2_raw_*` | Raw S2 features, downsampled/toy dataset |
| `*_smallfields` | Filtered to small fields only, using 7-class label scheme |
| `*_fewshot` | Few-shot training regime |
| `*_multirun` | 20-repeat evaluation with confidence intervals |
| `*_tune*` | Hyperparameter tuning experiments |

## Data Preprocessing Scripts

- `prep_s2_vis.py` — Computes NDVI, GCVI, EVI, LSWI from S2 bands and appends them
- `prep_s1_vis.py` — Computes RVI from S1 VV/VH and appends it
- `reduce_npy.py` — Spatial decimation of large `.npy` arrays (for fast prototyping)
- `remap_cropcodes_forsmallfields.py` — Remaps 17-class labels to 7-class small-fields scheme

## S2 Normalization Constants

```python
S2_BAND_MEAN = [1711.09, 1308.85, 1546.45, 3010.13, 3106.51, 2068.30, 2685.08, 2931.59, 2514.69, 1899.49]
S2_BAND_STD  = [1926.10, 1862.98, 1803.18, 1741.78, 1677.45, 1888.79, 1736.31, 1715.81, 1514.52, 1398.48]
S1_BAND_MEAN = [5484.04, 3003.78]
S1_BAND_STD  = [1871.23, 1726.07]
```

## MLP Architecture

3-layer MLP with BatchNorm + ReLU + Dropout after each hidden layer. Default hidden sizes vary by script. Trained with cross-entropy loss, Adam optimizer, early stopping on validation loss (patience=5). Class imbalance handled via `compute_class_weight`.

## Summary Table Generation

`modelsummarymultirun/create_summary_table.py` — aggregates per-treatment multirun CSV results into a multi-index summary table (mean ± 95% CI per model per metric).

### Environment Note
- The Python executable is accessed via the `py` command. 
- Always use `py` instead of `python` for running scripts, installing packages, or running tests.
- Example: `py -m pytest` or `py script.py`

## Execution Note
    If the refactored code does not exactly replicate the funcionality of the original code, it is considered a fail.
