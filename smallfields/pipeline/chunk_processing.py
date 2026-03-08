from __future__ import annotations
import numpy as np
from smallfields.data.features import FeatureLoader


def generate_chunks(H: int, W: int, chunk_size: int) -> list:
    """Generate (h_start, h_end, w_start, w_end) chunk tuples."""
    # min() clamps the end boundary so the last chunk doesn't exceed the array edge
    return [
        (h, min(h + chunk_size, H), w, min(w + chunk_size, W))
        for h in range(0, H, chunk_size)
        for w in range(0, W, chunk_size)
    ]


def process_chunk(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    labels: np.ndarray,
    field_ids: np.ndarray,
    valid_classes: set,
    train_fids,
    val_fids,
    test_fids,
    feature_loader: FeatureLoader,
) -> tuple:
    """Load features and split a spatial chunk into train/val/test pixel sets.

    Parameters
    ----------
    h_start, h_end, w_start, w_end : int
        Chunk boundaries.
    labels : np.ndarray, shape (H, W)
    field_ids : np.ndarray, shape (H, W)
    valid_classes : set
    train_fids, val_fids, test_fids : array-like
    feature_loader : FeatureLoader

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) — all np.ndarray
    """
    # Load features for this chunk: (h*w, num_features)
    X_chunk = feature_loader(h_start, h_end, w_start, w_end)

    # Flatten the spatial dims to match the feature row order (row-major)
    y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
    fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()

    # Remove background / discarded pixels — only valid classes enter training/eval.
    # np.isin is O(n * len(valid_classes)) but faster than a Python loop.
    valid_mask = np.isin(y_chunk, list(valid_classes))
    X_chunk = X_chunk[valid_mask]
    y_chunk = y_chunk[valid_mask]
    fieldid_chunk = fieldid_chunk[valid_mask]

    # Assign each pixel to a split based on which field it belongs to.
    # Field-level splitting prevents data leakage: all pixels of a field go to
    # the same split, so the classifier never sees training and test pixels from
    # the same physical field.
    train_mask = np.isin(fieldid_chunk, train_fids)
    val_mask = np.isin(fieldid_chunk, val_fids)
    test_mask = np.isin(fieldid_chunk, test_fids)

    return (
        X_chunk[train_mask], y_chunk[train_mask],
        X_chunk[val_mask],   y_chunk[val_mask],
        X_chunk[test_mask],  y_chunk[test_mask],
    )


def combine_chunk_results(results: list) -> tuple:
    """Stack per-chunk (X_train, y_train, ...) tuples into full arrays.

    Empty chunks (size 0) are skipped.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) — all np.ndarray
    """
    def _vstack(idx):
        # Collect non-empty feature matrices and vertically stack them
        parts = [r[idx] for r in results if r[idx].size > 0]
        return np.vstack(parts) if parts else np.empty((0,))

    def _hstack(idx):
        # Collect non-empty label vectors and horizontally concatenate them
        parts = [r[idx] for r in results if r[idx].size > 0]
        return np.hstack(parts) if parts else np.empty((0,))

    # Indices 0, 2, 4 are feature matrices (X); indices 1, 3, 5 are label vectors (y)
    X_train = _vstack(0)
    y_train = _hstack(1)
    X_val   = _vstack(2)
    y_val   = _hstack(3)
    X_test  = _vstack(4)
    y_test  = _hstack(5)

    return X_train, y_train, X_val, y_val, X_test, y_test
