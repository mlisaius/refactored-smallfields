from __future__ import annotations
import numpy as np
from smallfields.data.features import FeatureLoader


def batch_predict_chunk(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    labels: np.ndarray,
    field_ids: np.ndarray,
    valid_classes: set,
    train_fids,
    model,
    feature_loader: FeatureLoader,
    inner_batch_size: int = 1000,
) -> tuple:
    """Predict labels for a spatial chunk.

    Training pixels are copied directly from ground truth.
    Non-training valid pixels are predicted by the model.

    Returns
    -------
    (h_start, h_end, w_start, w_end, chunk_pred)
    """
    chunk_labels = labels[h_start:h_end, w_start:w_end]
    chunk_fieldids = field_ids[h_start:h_end, w_start:w_end]
    # Initialise with zeros (background); only valid pixels will be overwritten
    chunk_pred = np.zeros_like(chunk_labels)

    # Identify which pixels are valid (non-background) and which are training pixels
    valid_mask = np.isin(chunk_labels, list(valid_classes))
    non_train_mask = ~np.isin(chunk_fieldids, train_fids)
    # Predict only pixels that are both valid and NOT in the training set
    predict_mask = valid_mask & non_train_mask

    # Training pixels are written directly from ground-truth labels.
    # This matches the source scripts, which paint training pixels onto the
    # prediction map before writing model predictions for the rest.
    train_mask = valid_mask & ~non_train_mask
    chunk_pred[train_mask] = chunk_labels[train_mask]

    if not np.any(predict_mask):
        # No pixels to predict in this chunk (e.g. chunk is entirely training or background)
        return h_start, h_end, w_start, w_end, chunk_pred

    # Get the (row, col) coordinates of pixels that need model predictions
    h_indices, w_indices = np.where(predict_mask)

    # Load features once for the whole chunk: shape (h*w, num_features)
    # Row-major order: pixel at (i, j) within the chunk maps to flat index i*w_size + j
    features_all = feature_loader(h_start, h_end, w_start, w_end)
    h_size = h_end - h_start
    w_size = w_end - w_start
    # features_all[i*w_size + j] corresponds to pixel (i, j) in the chunk
    flat_indices = h_indices * w_size + w_indices

    # Run model in inner batches to avoid OOM on large chunks
    for i in range(0, len(flat_indices), inner_batch_size):
        batch_idx = flat_indices[i : i + inner_batch_size]
        batch_features = features_all[batch_idx]
        batch_preds = model.predict(batch_features)
        batch_h = h_indices[i : i + inner_batch_size]
        batch_w = w_indices[i : i + inner_batch_size]
        # Write predictions at their original (row, col) positions in the chunk
        chunk_pred[batch_h, batch_w] = batch_preds

    return h_start, h_end, w_start, w_end, chunk_pred


def build_prediction_map(
    pred_results: list,
    H: int,
    W: int,
) -> np.ndarray:
    """Assemble per-chunk prediction results into a full (H, W) map."""
    pred_map = np.zeros((H, W), dtype=np.int64)
    # Each result tuple contains the chunk's position and its prediction sub-array;
    # writing each sub-array back into the correct slice reconstructs the full map.
    for h_start, h_end, w_start, w_end, chunk_pred in pred_results:
        pred_map[h_start:h_end, w_start:w_end] = chunk_pred
    return pred_map
