# refactored-smallfields

A clean, modular Python package for classifying crop types in small Austrian agricultural fields using satellite imagery. This is a refactor of a collection of research scripts into something you can actually reuse without copy-pasting code everywhere.

---

## What's the point?

The original research compared two kinds of features for crop classification:
- **BTFM embeddings** — representations from a pre-trained Barlow Twins Foundation Model (TESSERA)
- *Raw satellite bands** — Sentinel-2 optical bands and Sentinel-1 SAR (radar) bands, with optional vegetation indices

Fields are classified into either 7 or 17 crop types. The tricky part is that the fields are *small*, so there aren't many pixels per field — this is what makes it interesting and hard.

---

## What's in here

```
smallfields/          ← the actual Python package
  config.py           ← class names, normalization constants, config dataclasses
  data/
    loading.py        ← load .npy arrays, filter by distance, identify valid classes
    splitting.py      ← area-weighted and few-shot field splits
    features.py       ← load and normalize feature chunks (BTFM, S2, S1, VIs)
  models/
    classifiers.py    ← build/train RF, LR, SVM, XGBoost
    mlp.py            ← 3-layer MLP with early stopping (PyTorch)
  pipeline/
    chunk_processing.py  ← split image into tiles, assign pixels to train/val/test
    prediction.py        ← run model over the full image, build prediction map
  evaluation/
    metrics.py        ← pixel accuracy, per-class accuracy, confidence intervals
    visualization.py  ← confusion matrix, per-class bar chart, classification map

scripts/              ← CLI scripts you actually run
  run_experiment.py   ← main entry point for running experiments
  tune_hyperparams.py ← grid search over hyperparameters
  prep_s2_vis.py      ← compute NDVI/GCVI/EVI/LSWI and append to S2 array
  prep_s1_vis.py      ← compute RVI and append to SAR array
  reduce_npy.py       ← spatially decimate arrays for quick prototyping
  remap_cropcodes.py  ← convert 17-class labels to 7-class small-fields scheme
  summarize_results.py ← aggregate multirun CSVs into a summary table
```

---

## Dependencies

Standard scientific Python stack — no special install steps:

```
numpy pandas scikit-learn xgboost torch matplotlib seaborn scipy joblib
```

Scripts are run directly with `python scripts/<script>.py`.

---

## Data format

Everything is stored as `.npy` arrays:

| File | Shape | Description |
|------|-------|-------------|
| `bands.npy` | `(T, H, W, 10)` | Sentinel-2 reflectance bands |
| `bands_VIs.npy` | `(T, H, W, 14)` | S2 bands + NDVI, GCVI, EVI, LSWI |
| `sar_ascending_VIs.npy` | `(T, H, W, 3)` | SAR ascending: VV, VH, RVI |
| `sar_descending_VIs.npy` | `(T, H, W, 3)` | SAR descending: VV, VH, RVI |
| `representations_*.npy` | `(H, W, C)` | BTFM embeddings |
| `fieldtype_7classes_*.npy` | `(H, W)` | Crop labels (1–7, 0=discard) |
| `fieldid.npy` | `(H, W)` | Field ID per pixel |
| `fielddata_*.csv` | — | Field metadata (area, crop code, field ID) |

Labels are 1-indexed (class 0 = discard/background). XGBoost is the exception — it gets 0-indexed labels automatically.

---

## Running an experiment

The main script is `scripts/run_experiment.py`. It handles single runs, 20-repeat multirun, and few-shot modes.

**BTFM + MLP, 20-repeat multirun:**
```bash
python scripts/run_experiment.py \
  --feature-source btfm \
  --model MLP \
  --mode multirun \
  --num-repeats 20 \
  --class-scheme 7class \
  --label-file /path/to/fieldtype_7classes.npy \
  --field-id-file /path/to/fieldid.npy \
  --field-data-csv /path/to/fielddata.csv \
  --output-dir /path/to/results/ \
  --btfm-representations-file /path/to/representations.npy
```

**Raw S2+S1+VIs + RandomForest, single run, small fields only:**
```bash
python scripts/run_experiment.py \
  --feature-source raw_s2s1vis \
  --model RandomForest \
  --mode single \
  --label-file /path/to/labels.npy \
  --field-id-file /path/to/fieldid.npy \
  --field-data-csv /path/to/fielddata.csv \
  --output-dir /path/to/results/ \
  --s2-bands-file /path/to/bands_VIs.npy \
  --sar-asc-file /path/to/sar_ascending_VIs.npy \
  --sar-desc-file /path/to/sar_descending_VIs.npy \
  --cloud-mask-file /path/to/masks.npy \
  --max-dist-csv /path/to/maxdist.csv \
  --max-dist-filter 3.0 \
  --max-dist-operator lte \
  --rf-n-estimators 200 \
  --rf-max-depth 50 \
  --rf-min-samples-split 5 \
  --rf-max-features none
```

### Model choices

Pass one of these to `--model`:
- `MLP` — 3-layer MLP with BatchNorm, ReLU, Dropout, early stopping
- `RandomForest`
- `LogisticRegression`
- `SVM`
- `XGBoost`

### Feature source choices

Pass one of these to `--feature-source`:
- `btfm` — BTFM embedding vectors (requires `--btfm-representations-file`)
- `raw_s2` — Sentinel-2 reflectance bands only (requires `--s2-bands-file`)
- `raw_s2s1` — S2 + S1 bands without VIs (requires `--s2-bands-file`, `--sar-asc-file`, `--sar-desc-file`)
- `raw_s2s1vis` — S2+VIs + S1+RVI (same as above, but with pre-computed VIs in the arrays)

### Output files

Results land in `--output-dir`:
- `<prefix>_run_results.csv` — one row per seed: `random_seed, validation_accuracy, test_accuracy, balanced_f1, macro_f1`
- `<prefix>_allmodels_summary_stats.csv` — mean ± 95% CI per metric (multirun only)
- `<prefix>_seed<N>_confusion_matrix.png`
- `<prefix>_seed<N>_per_class_accuracy.png`
- `<prefix>_classification_map.png` + `_pred_map.npy` (single-run only)

---

## Hyperparameter tuning

```bash
python scripts/tune_hyperparams.py \
  --feature-source btfm \
  --model MLP \
  --label-file /path/labels.npy \
  --field-id-file /path/fieldid.npy \
  --field-data-csv /path/fielddata.csv \
  --output-dir /path/results/ \
  --btfm-representations-file /path/representations.npy \
  --hidden-size-options "[512,256,128]" "[1024,512,256]" "[2048,1024,512]" \
  --dropout-options 0.3 0.4 0.5 \
  --lr-options 0.001 0.0005
```

Runs a full Cartesian product over all option combinations (seed=0 each time) and saves one CSV row per combination. Works for MLP, RandomForest, XGBoost, and SVM (SVM uses `GridSearchCV` instead).

---

## Data prep scripts

If you're starting from raw S2/S1 arrays without VIs pre-computed, run these first:

```bash
# Add NDVI, GCVI, EVI, LSWI to S2 bands
python scripts/prep_s2_vis.py --input bands.npy --output bands_VIs.npy

# Add RVI to SAR data (run for ascending and descending separately)
python scripts/prep_s1_vis.py --input sar_ascending.npy --output sar_ascending_VIs.npy
python scripts/prep_s1_vis.py --input sar_descending.npy --output sar_descending_VIs.npy

# Decimate arrays 10x spatially for quick tests
python scripts/reduce_npy.py \
  --input representations.npy --output representations_dec10.npy \
  --decimation-factor 10 --data-type representations

# Convert 17-class labels to 7-class small-fields scheme
python scripts/remap_cropcodes.py \
  --input fieldtype_17classes.npy \
  --output fieldtype_7classes_maddysmallfields.npy
```

---

## Aggregating multirun results

After running multiple experiments (different models or feature sources), collect everything into one summary table:

```bash
python scripts/summarize_results.py \
  --input-dir /path/to/results/ \
  --output-csv /path/to/results/summary_table.csv
```

It picks up all `*allmodels_summary_stats.csv` files in the directory and produces a wide table with mean ± 95% CI per model per metric, grouped by treatment (feature source).

---

## A few things worth knowing

- **Field-level splitting** — pixels are assigned to train/val/test based on which field they belong to, not randomly. This prevents data leakage between splits since all pixels of a field go to the same split.
- **Area-weighted greedy split** — for each crop class, the smallest fields are added to the training set first until 10% of the class area is reached. This maximises the number of training fields while keeping the area budget small.
- **MLP class weights** — on by default (helps with class imbalance). Use `--no-use-class-weights` to disable, which matches some of the original single-run scripts.
- **XGBoost labels** — XGBoost gets 0-indexed labels automatically; you don't need to do anything special.
- **Random seeds** — the legacy `numpy.random.seed` / `numpy.random.shuffle` API is used intentionally to reproduce the original script results exactly. Using the newer `numpy.random.default_rng` would give different field assignments for the same seed.
