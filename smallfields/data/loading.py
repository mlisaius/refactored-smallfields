from __future__ import annotations
import numpy as np
import pandas as pd
from collections import Counter


def load_labels_and_fields(label_file: str, field_id_file: str, zero_index_labels: bool = False):
    """Load label and field-ID arrays.

    Parameters
    ----------
    label_file : str
        Path to the .npy labels array (H, W).
        Values are 1-indexed crop classes (1–7 or 1–17); 0 = discard.
    field_id_file : str
        Path to the .npy field-ID array (H, W).
        Each pixel stores the integer ID of the field it belongs to;
        background / non-field pixels have ID 0.
    zero_index_labels : bool
        If True, subtract 1 from every label (used for XGBoost which requires
        0-indexed targets). After the shift the discard class becomes -1.

    Returns
    -------
    labels : np.ndarray, shape (H, W), dtype int64
    field_ids : np.ndarray, shape (H, W)
    H : int
    W : int
    """
    labels = np.load(label_file).astype(np.int64)  # cast to signed int so -1 is representable
    if zero_index_labels:
        # XGBoost expects labels in [0, num_classes). Shift 1-indexed labels
        # down by 1; the discard class (0) becomes -1 and is filtered later.
        labels -= 1
    field_ids = np.load(field_id_file)
    H, W = labels.shape
    return labels, field_ids, H, W


def load_field_data(field_data_csv: str, filtered_fids=None) -> pd.DataFrame:
    """Load field metadata CSV, optionally filtering to a set of field IDs.

    Parameters
    ----------
    field_data_csv : str
        Path to the CSV (fielddata_smallfieldlabelgroups.csv).
        Key columns: fid_1 (field ID), remapped_SNAR_CODE (7-class crop code),
        SNAR_CODE (17-class), area_m2 (field area in square metres).
    filtered_fids : array-like or None
        If provided, keep only rows whose 'fid_1' is in this set.
        Used after max-distance filtering to restrict the field pool.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(field_data_csv)
    if filtered_fids is not None:
        # Retain only the fields that passed the distance filter so that
        # train/val/test splits are drawn exclusively from eligible fields.
        df = df[df["fid_1"].isin(filtered_fids)]
    return df


def apply_max_dist_filter(field_ids: np.ndarray, maxdist_csv: str, threshold: float, operator: str):
    """Filter field IDs based on a maximum-distance criterion.

    The max-distance CSV records the longest internal distance of each field
    (a proxy for field size / shape). Filtering to small fields (e.g.
    max_dist <= 3) reproduces the smallfields experimental condition.

    Parameters
    ----------
    field_ids : np.ndarray, shape (H, W)
        Original field-ID array.
    maxdist_csv : str
        CSV with at least two columns: 'fid_1' and 'max_dist'.
    threshold : float
        Distance threshold value.
    operator : str
        Comparison direction: 'lte' (≤), 'gte' (≥), 'lt' (<), 'gt' (>).

    Returns
    -------
    filtered_field_ids : np.ndarray, shape (H, W)
        field_ids with pixels that belong to excluded fields set to 0.
    filtered_fids_set : set
        Set of field IDs that passed the filter (used to restrict the CSV).
    """
    # np.genfromtxt with names=True reads the header row as column names,
    # avoiding a pandas dependency here and matching the source exactly.
    maxdist = np.genfromtxt(maxdist_csv, delimiter=",", names=True, dtype=None, encoding=None)
    ops = {
        "lte": lambda a, b: a <= b,
        "gte": lambda a, b: a >= b,
        "lt":  lambda a, b: a < b,
        "gt":  lambda a, b: a > b,
    }
    if operator not in ops:
        raise ValueError(f"Unknown operator '{operator}'. Use one of: {list(ops)}")

    # Boolean mask over the CSV rows
    mask = ops[operator](maxdist["max_dist"], threshold)
    filtered_fids = maxdist["fid_1"][mask]
    filtered_fids_set = set(filtered_fids.tolist())

    # Zero-out any pixel that belongs to an excluded field; background pixels
    # (already 0) remain 0 and are naturally ignored downstream.
    filtered_field_ids = np.where(np.isin(field_ids, filtered_fids), field_ids, 0)
    return filtered_field_ids, filtered_fids_set


def identify_valid_classes(labels: np.ndarray, background_value: int, min_count: int = 2) -> set:
    """Return the set of class values that appear at least `min_count` times.

    Classes with fewer than 2 pixels are excluded to ensure at least one
    training and one test pixel exists — matching the source scripts.

    Parameters
    ----------
    labels : np.ndarray
        Label array (any shape).
    background_value : int
        Class value to discard.
        Use 0 for 1-indexed labels (non-XGBoost) or -1 for XGBoost
        (labels already shifted by -1, so 0→-1 is the discard class).
    min_count : int
        Minimum pixel count to include a class (default 2, matches source).

    Returns
    -------
    set of int
    """
    class_counts = Counter(labels.ravel())
    valid = {cls for cls, count in class_counts.items() if count >= min_count}
    # Remove the background/discard class so it never enters training or eval
    valid.discard(background_value)
    return valid
