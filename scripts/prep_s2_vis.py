"""Compute vegetation indices from Sentinel-2 bands and append them.

Source: source-temp/prep_s2_vis.py

Reads a (T, H, W, B) array with at least 10 S2 bands and appends four
vegetation indices (NDVI, GCVI, EVI, LSWI), producing (T, H, W, B+4).

Band index mapping (0-based):
  0 = Red, 1 = Blue, 2 = Green, 3 = NIR, 8 = SWIR16

Usage
-----
python scripts/prep_s2_vis.py --input bands.npy --output bands_VIs.npy
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Append vegetation indices to S2 bands array.")
    p.add_argument("--input", required=True, help="Path to input bands .npy (T,H,W,B).")
    p.add_argument("--output", required=True, help="Path for output bands_VIs .npy.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(f"Loading S2 data from {args.input}")
    s2_data = np.load(args.input)  # shape: (T, H, W, B)
    logging.info(f"Loaded data with shape {s2_data.shape}")

    # Extract individual bands by their 0-based column index — matches source exactly
    red    = s2_data[:, :, :, 0]   # Red   (index 0)
    blue   = s2_data[:, :, :, 1]   # Blue  (index 1) — loaded but not used in EVI formula
    green  = s2_data[:, :, :, 2]   # Green (index 2)
    nir    = s2_data[:, :, :, 3]   # NIR   (index 3)
    swir16 = s2_data[:, :, :, 8]   # SWIR16 (index 8, 1600 nm band)

    # 1. NDVI — Normalised Difference Vegetation Index
    #    Range [-1, 1]; high values indicate healthy vegetation
    ndvi = (nir - red) / (nir + red)

    # 2. GCVI — Green Chlorophyll Vegetation Index
    #    Sensitive to chlorophyll content; uses NIR and green bands
    gcvi = (nir / green) - 1

    # 3. EVI — Enhanced Vegetation Index (matches source formula exactly)
    #    The source does NOT include a blue correction term (coefficient is 0):
    #    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red + 1)
    #    Note: standard EVI formula is EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    #    The source omits the blue term, so we replicate that here.
    evi = 2.5 * ((nir - red) / (nir + 6 * red + 1))

    # 4. LSWI — Land Surface Water Index
    #    Sensitive to soil/vegetation water content; uses NIR and SWIR16
    lswi = (nir - swir16) / (nir + swir16)

    # Replace NaN / ±Inf (from zero denominators) with 0 — matches source nan_to_num calls
    ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    gcvi = np.nan_to_num(gcvi, nan=0.0, posinf=0.0, neginf=0.0)
    evi  = np.nan_to_num(evi,  nan=0.0, posinf=0.0, neginf=0.0)
    lswi = np.nan_to_num(lswi, nan=0.0, posinf=0.0, neginf=0.0)

    # Stack the four VIs as new bands → shape (T, H, W, B+4)
    # axis=-1 appends along the band dimension, consistent with (T, H, W, B) layout
    s2_with_vis = np.concatenate(
        [s2_data, np.stack([ndvi, gcvi, evi, lswi], axis=-1)], axis=-1
    )

    logging.info(f"New data shape with VIs: {s2_with_vis.shape}")
    np.save(args.output, s2_with_vis)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
