from __future__ import annotations
import numpy as np
from smallfields.config import NormConstants


class FeatureLoader:
    """Pickle-serializable callable that loads a spatial chunk of features.

    ``joblib.Parallel`` serialises its arguments with pickle before dispatching
    work to subprocesses.  A closure over ``bands_file`` would not be
    picklable, so feature loading is encapsulated in this class instead.

    Parameters
    ----------
    feature_source : str
        One of 'btfm', 'raw_s2', 'raw_s2s1', 'raw_s2s1vis'.
    kwargs : dict
        File paths and options forwarded to the appropriate loader function.
        Keys: representations_file, bands_file, sar_asc_file, sar_desc_file,
              cloud_mask_file, norm.
    """

    def __init__(self, feature_source: str, **kwargs):
        self.feature_source = feature_source
        self.kwargs = kwargs

    def __call__(self, h_start: int, h_end: int, w_start: int, w_end: int) -> np.ndarray:
        """Return features for the spatial chunk, shape (h*w, num_features)."""
        loaders = {
            "btfm":        load_chunk_btfm,
            "raw_s2":      load_chunk_raw_s2,
            "raw_s2s1":    load_chunk_raw_s2s1,
            "raw_s2s1vis": load_chunk_raw_s2s1vis,
        }
        if self.feature_source not in loaders:
            raise ValueError(
                f"Unknown feature_source '{self.feature_source}'. "
                f"Choose from: {list(loaders)}"
            )
        return loaders[self.feature_source](h_start, h_end, w_start, w_end, **self.kwargs)


# ---------------------------------------------------------------------------
# Individual loader functions
# ---------------------------------------------------------------------------

def load_chunk_btfm(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    representations_file: str,
    **_,  # absorb unused kwargs (norm, bands_file, etc.) without error
) -> np.ndarray:
    """Load BTFM representations for a chunk.

    The BTFM file has shape (H, W, C) where C is the embedding dimension
    (e.g. 768 for a ViT-Base model).  No normalisation is applied — the
    embeddings are already in a learned representation space.

    Returns: (h*w, C)
    """
    # Load only the required spatial slice; avoids loading the full array
    chunk = np.load(representations_file)[h_start:h_end, w_start:w_end, :]
    h, w, c = chunk.shape
    # Flatten spatial dims: each row = one pixel's embedding vector
    return chunk.reshape(-1, c)


def load_chunk_raw_s2(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    bands_file: str,
    norm: NormConstants = None,
    cloud_mask_file: str = "",
    **_,
) -> np.ndarray:
    """Load and normalise raw Sentinel-2 bands for a chunk.

    Bands file shape: (T, H, W, B) — T time steps, B=10 spectral bands.
    Normalisation: z-score per band using the dataset-wide statistics in
    ``NormConstants`` (broadcast over T, H, W dimensions).
    Cloud mask: binary (T, H, W) array multiplied element-wise; masked
    observations become 0 in normalised space (a neutral value).

    The spatial and band dimensions are transposed so that the feature vector
    for each pixel is the concatenation of its B values at each of the T time
    steps: layout is [b0_t0, b1_t0, …, bB_t0, b0_t1, …, bB_tT-1].

    Returns: (h*w, T*B)
    """
    if norm is None:
        norm = NormConstants()

    # (T, h, w, B) slice from the full array
    tile_chunk = np.load(bands_file)[:, h_start:h_end, w_start:w_end, :]

    # Z-score normalisation; S2_BAND_MEAN/STD broadcast over (T, h, w) dims
    tile_chunk = (tile_chunk - norm.S2_BAND_MEAN) / norm.S2_BAND_STD

    if cloud_mask_file:
        # mask shape (T, h, w) → expand to (T, h, w, 1) for broadcasting
        s2_mask = np.load(cloud_mask_file)[:, h_start:h_end, w_start:w_end]
        tile_chunk = tile_chunk * s2_mask[..., np.newaxis]

    T, h, w, B = tile_chunk.shape
    # Transpose to (h, w, T, B) then reshape to (h*w, T*B) so that each
    # row in the output matrix is a single pixel's full time-series feature vector
    return tile_chunk.transpose(1, 2, 0, 3).reshape(-1, T * B)


def load_chunk_raw_s2s1(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    bands_file: str,
    sar_asc_file: str,
    sar_desc_file: str,
    norm: NormConstants = None,
    cloud_mask_file: str = "",
    **_,
) -> np.ndarray:
    """Load and normalise raw S2 + S1 bands for a chunk (no vegetation indices).

    S2 file shape: (T_s2, H, W, 10) — 10 reflectance bands per time step.
    SAR files shape: (T_s1, H, W, 2) — [VV, VH] per time step.
    Ascending and descending SAR passes are concatenated along the time axis
    (axis=0), giving a combined SAR array of shape (2*T_s1, H, W, 2).

    Both S2 and S1 bands are z-score normalised with their respective
    dataset-wide statistics.  The normalised arrays are then flattened and
    concatenated, giving the feature layout:
      [S2_t0_b0..b9, S2_t1_b0..b9, …, S1_asc_t0_vv_vh, …, S1_desc_t0_vv_vh, …]

    Returns: (h*w, T_s2*10 + 2*T_s1*2)
    """
    if norm is None:
        norm = NormConstants()

    # --- Sentinel-2 ---
    s2 = np.load(bands_file)[:, h_start:h_end, w_start:w_end, :]
    s2 = (s2 - norm.S2_BAND_MEAN) / norm.S2_BAND_STD
    if cloud_mask_file:
        s2_mask = np.load(cloud_mask_file)[:, h_start:h_end, w_start:w_end]
        s2 = s2 * s2_mask[..., np.newaxis]
    T_s2, h, w, B_s2 = s2.shape
    s2_flat = s2.transpose(1, 2, 0, 3).reshape(-1, T_s2 * B_s2)

    # --- Sentinel-1 (ascending + descending concatenated along time) ---
    sar_asc = np.load(sar_asc_file)[:, h_start:h_end, w_start:w_end]
    sar_desc = np.load(sar_desc_file)[:, h_start:h_end, w_start:w_end]
    # Concatenate on axis=0 (time) so ascending passes come first
    sar = np.concatenate((sar_asc, sar_desc), axis=0)
    # Normalise VV and VH with the same S1 statistics applied to the
    # combined ascending+descending array (matches source process_chunk)
    sar = (sar - norm.S1_BAND_MEAN) / norm.S1_BAND_STD
    T_s1, h, w, B_s1 = sar.shape
    sar_flat = sar.transpose(1, 2, 0, 3).reshape(-1, T_s1 * B_s1)

    # Concatenate S2 and S1 feature vectors per pixel
    return np.concatenate((s2_flat, sar_flat), axis=1)


def load_chunk_raw_s2s1vis(
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    bands_file: str,
    sar_asc_file: str,
    sar_desc_file: str,
    cloud_mask_file: str,
    norm: NormConstants = None,
    **_,
) -> np.ndarray:
    """Load S2+VIs + S1+RVI for a chunk with partial normalisation.

    Requires ``bands_file`` to already contain the vegetation indices appended
    by ``scripts/prep_s2_vis.py``, and SAR files to contain RVI from
    ``scripts/prep_s1_vis.py``.

    S2 file shape: (T, H, W, 14)  — first 10 cols = reflectance bands,
                                    last 4 cols = NDVI, GCVI, EVI, LSWI.
    SAR files shape: (T, H, W, 3) — first 2 cols = VV, VH;  last 1 = RVI.

    Normalisation strategy (matches source austria_s1s2_rf_smallfields.py):
    - S2 reflectance bands (indices 0–9): z-score with S2_BAND_MEAN/STD.
    - S2 VIs (indices 10–13): left as-is (already in [-1, 1] or similar).
    - S1 VV/VH (indices 0–1): z-score with S1_BAND_MEAN/STD.
    - S1 RVI (index 2): left as-is.
    Cloud mask is applied AFTER normalisation so that masked pixels become 0.

    Returns: (h*w, T_s2*14 + 2*T_s1*3)
    """
    if norm is None:
        norm = NormConstants()

    # --- Sentinel-2 + vegetation indices ---
    s2_data = np.load(bands_file)[:, h_start:h_end, w_start:w_end, :]
    s2_bands = s2_data[..., :10]    # reflectance bands only
    s2_vis   = s2_data[..., 10:]    # NDVI, GCVI, EVI, LSWI — not normalised
    s2_bands = (s2_bands - norm.S2_BAND_MEAN) / norm.S2_BAND_STD
    # Recombine normalised bands with raw VIs
    s2_data = np.concatenate([s2_bands, s2_vis], axis=-1)

    # Apply cloud mask: 0 in mask → 0 in feature (neutral in normalised space)
    s2_mask = np.load(cloud_mask_file)[:, h_start:h_end, w_start:w_end]
    s2_data = s2_data * s2_mask[..., np.newaxis]

    T_s2, h, w, B_s2 = s2_data.shape
    s2_flat = s2_data.transpose(1, 2, 0, 3).reshape(-1, T_s2 * B_s2)

    # --- Sentinel-1 + RVI ---
    sar_asc  = np.load(sar_asc_file)[:, h_start:h_end, w_start:w_end]
    sar_desc = np.load(sar_desc_file)[:, h_start:h_end, w_start:w_end]
    sar = np.concatenate((sar_asc, sar_desc), axis=0)  # merge along time
    sar_bands = sar[..., :2]   # VV, VH — to be normalised
    sar_rvi   = sar[..., 2:]   # RVI — not normalised (bounded ratio index)
    sar_bands = (sar_bands - norm.S1_BAND_MEAN) / norm.S1_BAND_STD
    sar = np.concatenate([sar_bands, sar_rvi], axis=-1)

    T_s1, h, w, B_s1 = sar.shape
    sar_flat = sar.transpose(1, 2, 0, 3).reshape(-1, T_s1 * B_s1)

    return np.concatenate((s2_flat, sar_flat), axis=1)
