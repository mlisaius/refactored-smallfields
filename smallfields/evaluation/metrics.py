from __future__ import annotations
import numpy as np
from scipy import stats


def compute_pixel_accuracy(
    labels: np.ndarray,
    pred_map: np.ndarray,
    field_ids: np.ndarray,
    valid_classes: set,
    target_fids,
) -> float:
    """Compute pixel-level accuracy for pixels belonging to target_fids.

    Replaces the O(H*W) nested Python loop in the source with vectorised numpy.

    Parameters
    ----------
    labels : np.ndarray, shape (H, W)
    pred_map : np.ndarray, shape (H, W)
    field_ids : np.ndarray, shape (H, W)
    valid_classes : set
    target_fids : array-like

    Returns
    -------
    float — accuracy in percent, or 0.0 if no pixels found.
    """
    # Build a combined mask: pixel must be a valid class AND belong to a target field
    valid_mask = np.isin(labels, list(valid_classes))
    fid_mask = np.isin(field_ids, target_fids)
    combined = valid_mask & fid_mask
    n_pixels = combined.sum()
    if n_pixels == 0:
        return 0.0
    # Count pixels where the prediction matches the ground-truth label
    n_correct = (pred_map[combined] == labels[combined]).sum()
    return float(n_correct / n_pixels * 100)


def compute_per_class_accuracy(
    labels: np.ndarray,
    pred_map: np.ndarray,
    field_ids: np.ndarray,
    valid_classes: set,
    target_fids,
) -> dict:
    """Per-class pixel accuracy for pixels belonging to target_fids.

    Returns
    -------
    dict mapping class_id -> accuracy (%) for classes with > 0 test pixels.
    """
    fid_mask = np.isin(field_ids, target_fids)
    class_accuracies = {}
    for cls in sorted(valid_classes):
        # Isolate pixels of this class that belong to the target (e.g. test) fields
        cls_mask = (labels == cls) & fid_mask
        n_pixels = cls_mask.sum()
        if n_pixels > 0:
            n_correct = (pred_map[cls_mask] == cls).sum()
            class_accuracies[cls] = float(n_correct / n_pixels * 100)
    return class_accuracies


def compute_mean_ci(series, confidence: float = 0.95) -> tuple:
    """Compute mean and 95% confidence interval half-width."""
    data = np.array(series)
    n = len(data)
    mean = np.mean(data)
    # Standard error of the mean
    se = stats.sem(data)
    # t critical value for a two-tailed interval with (n-1) degrees of freedom
    h = se * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return float(mean), float(h)


def collect_run_metrics(
    seed: int,
    val_accuracy: float,
    test_accuracy: float,
    f1_weighted: float,
    f1_macro: float,
) -> dict:
    """Assemble a single-run metrics dict.

    Column names and scaling match the source multirun CSV format exactly:
      random_seed, validation_accuracy, test_accuracy, balanced_f1, macro_f1

    balanced_f1 = weighted-average F1 * 100  (matches source naming convention)
    macro_f1    = macro-average F1 * 100
    """
    return {
        "random_seed": seed,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        # 'balanced_f1' in the source is actually the weighted-avg F1 (not balanced/macro)
        "balanced_f1": f1_weighted * 100,
        "macro_f1": f1_macro * 100,
    }
