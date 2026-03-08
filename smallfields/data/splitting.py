from __future__ import annotations
import numpy as np
import pandas as pd


def area_weighted_split(
    fielddata_df: pd.DataFrame,
    snar_code_column: str,
    fid_column: str,
    area_column: str,
    training_ratio: float,
    val_test_split_ratio: float,
    seed: int,
) -> tuple:
    """Split fields into train / val / test sets using area-weighted greedy selection.

    Algorithm (matches source scripts exactly):

    1. For each crop class (SNAR code), compute total area of all fields.
    2. Sort fields ascending by area (smallest first).
    3. Greedily add fields to the training set until the cumulative area
       reaches ``training_ratio * total_class_area``.  The first field that
       would push cumulative area over the threshold is NOT added (the loop
       breaks on ``>=`` not ``>``), so the training set slightly undershoots.
    4. Deduplicate training IDs (a field could theoretically match multiple
       classes if the CSV has duplicates).
    5. The remaining fields are shuffled with ``np.random.seed(seed)``
       (legacy Mersenne-Twister — must match the ``np.random.seed(seed)``
       call in the source multirun loops).
    6. The first ``int(len(remaining) * val_test_split_ratio)`` shuffled
       fields go to validation; the rest go to test.

    Parameters
    ----------
    fielddata_df : pd.DataFrame
        Field metadata. Must contain snar_code_column, fid_column, area_column.
    snar_code_column : str
        Column holding the crop-type code used for class-stratified splitting.
    fid_column : str
        Column holding the integer field ID.
    area_column : str
        Column holding field area in m².
    training_ratio : float
        Fraction of total per-class area to use for training (default 0.1).
    val_test_split_ratio : float
        Fraction of remaining fields allocated to validation (default 1/7).
    seed : int
        Random seed. In multirun mode the outer loop passes seed=0,1,…,N-1
        to produce different splits per repeat.

    Returns
    -------
    train_fids : np.ndarray
    val_fids : np.ndarray
    test_fids : np.ndarray

    Notes
    -----
    The legacy ``np.random.seed`` + ``np.random.shuffle`` API (MT19937) is
    used intentionally.  The source scripts call ``np.random.seed(seed)``
    then immediately ``np.random.shuffle(remaining)``, so the shuffle output
    is the first random draw from the seeded MT state.  Using
    ``np.random.default_rng`` (PCG64) would produce different field
    assignments for the same seed value.
    """
    # Compute total area per crop class for the greedy threshold
    area_summary = (
        fielddata_df.groupby(snar_code_column)[area_column]
        .sum()
        .reset_index()
        .rename(columns={area_column: "total_area"})
    )

    train_fids = []
    for _, row in area_summary.iterrows():
        sn_code = row[snar_code_column]
        total_area = row["total_area"]
        target_area = total_area * training_ratio  # area budget for this class

        # Sort ascending so the smallest fields enter training first,
        # maximising the number of fields while respecting the area budget.
        rows_sncode = (
            fielddata_df[fielddata_df[snar_code_column] == sn_code]
            .sort_values(by=area_column)
        )
        selected_area_sum = 0
        for _, r2 in rows_sncode.iterrows():
            if selected_area_sum < target_area:
                train_fids.append(int(r2[fid_column]))
                selected_area_sum += r2[area_column]
            else:
                break  # area budget exhausted for this class

    # Deduplicate in case a field appears under multiple SNAR codes
    train_fids = list(set(train_fids))

    # Build the remaining (val+test) pool by set subtraction
    all_fields = fielddata_df[fid_column].unique().astype(int)
    remaining = np.array(list(set(all_fields) - set(train_fids)))

    # Shuffle with the legacy MT19937 RNG to match source scripts exactly.
    # The source does: np.random.seed(seed) → np.random.shuffle(remaining)
    # with NO other random calls in between, so the result is fully determined
    # by the seed.
    np.random.seed(seed)
    np.random.shuffle(remaining)

    # First fraction → validation; rest → test
    val_count = int(len(remaining) * val_test_split_ratio)
    val_fids = remaining[:val_count]
    test_fids = remaining[val_count:]
    train_fids = np.array(train_fids)

    return train_fids, val_fids, test_fids


def fewshot_split(
    fielddata_df: pd.DataFrame,
    snar_code_column: str,
    fid_column: str,
    num_fields_per_class: int,
    seed: int,
) -> tuple:
    """Split fields for few-shot learning: N fields per class for train and val.

    Algorithm (matches ``austria_BTFM_mlp_smallfields_fewshot.py``):

    For each crop class, shuffle the available field IDs, then:
    - If the class has >= 2N fields: take the first N for train, next N for
      val, and all remaining for test.
    - Otherwise (small class): use ``min(10, len//2)`` for train, up to 10
      for val, and the rest for test.

    The MT19937 seed is set once before the class loop so that each per-class
    shuffle advances the same shared random state, matching the source.

    Parameters
    ----------
    fielddata_df : pd.DataFrame
    snar_code_column : str
    fid_column : str
    num_fields_per_class : int
        N fields per class for train (and val). Default 50 in source scripts.
    seed : int

    Returns
    -------
    train_fids : np.ndarray
    val_fids : np.ndarray
    test_fids : np.ndarray
    """
    # Set the global MT state once; each np.random.shuffle call advances it,
    # which matches the source script's sequential shuffle calls per class.
    np.random.seed(seed)

    train_fids = []
    val_fids = []
    test_fids = []

    snar_codes = fielddata_df[snar_code_column].unique()
    for sn_code in snar_codes:
        rows = fielddata_df[fielddata_df[snar_code_column] == sn_code]
        fids = rows[fid_column].unique()
        fids = np.array(fids)
        np.random.shuffle(fids)  # advances MT state; order matches source
        N = num_fields_per_class
        if len(fids) >= N * 2:
            # Enough fields for a clean N-train / N-val split
            train_fids.extend(fids[:N].tolist())
            val_fids.extend(fids[N:N * 2].tolist())
            test_fids.extend(fids[N * 2:].tolist())
        else:
            # Fallback for very small classes: use up to 10 fields each
            n_train = min(10, len(fids) // 2)
            n_val = min(10, len(fids) - n_train)
            train_fids.extend(fids[:n_train].tolist())
            val_fids.extend(fids[n_train:n_train + n_val].tolist())
            test_fids.extend(fids[n_train + n_val:].tolist())

    return np.array(train_fids), np.array(val_fids), np.array(test_fids)


def create_split_mask(
    field_ids: np.ndarray,
    train_fids,
    val_fids,
    test_fids,
) -> np.ndarray:
    """Create an (H, W) int8 mask: 1=train, 2=val, 3=test, 0=other.

    Replaces the O(H×W) Python loop in the source with vectorised np.isin.
    Used only for visualisation (the split map plot); not used in training.

    Parameters
    ----------
    field_ids : np.ndarray, shape (H, W)
    train_fids, val_fids, test_fids : array-like

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype int8
    """
    mask = np.zeros(field_ids.shape, dtype=np.int8)
    mask[np.isin(field_ids, train_fids)] = 1  # blue in visualisation
    mask[np.isin(field_ids, val_fids)] = 2    # green
    mask[np.isin(field_ids, test_fids)] = 3   # red
    return mask
