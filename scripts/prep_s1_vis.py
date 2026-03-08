"""Compute RVI from Sentinel-1 VV/VH bands and append it.

Source: source-temp/prep_s1_vis.py

Reads a (..., 2) SAR array (last dim = [VV, VH]) and appends RVI as a
third channel, producing (..., 3).

RVI = 4*VH / (VV + VH)

Usage
-----
python scripts/prep_s1_vis.py --input sar_ascending.npy --output sar_ascending_VIs.npy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Append RVI to SAR bands array.")
    p.add_argument("--input", required=True, help="Path to input SAR .npy (...,2).")
    p.add_argument("--output", required=True, help="Path for output .npy (...,3).")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(f"Loading S1 data from {args.input}")
    s1_data = np.load(args.input)  # shape: (..., 2) where last dim = [VV, VH]
    logging.info(f"Loaded data with shape {s1_data.shape}")

    # Extract VV and VH from the last dimension (index 0 and 1 respectively)
    vv = s1_data[..., 0]
    vh = s1_data[..., 1]

    # Radar Vegetation Index (RVI) — matches source exactly.
    # Range [0, 1]: higher values indicate denser vegetation canopies.
    # Formula: RVI = 4*VH / (VV + VH)
    rvi = (4 * vh) / (vv + vh)
    # Zero-out NaN and Inf that arise when VV + VH == 0 (e.g. masked/water pixels)
    rvi = np.nan_to_num(rvi, nan=0.0, posinf=0.0, neginf=0.0)

    # Append RVI as a new band: shape (..., 2) → (..., 3)
    # np.newaxis expands rvi from (...) to (..., 1) for concatenation along axis=-1
    s1_with_rvi = np.concatenate([s1_data, rvi[..., np.newaxis]], axis=-1)

    logging.info(f"New data shape with RVI: {s1_with_rvi.shape}")
    np.save(args.output, s1_with_rvi)
    logging.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
