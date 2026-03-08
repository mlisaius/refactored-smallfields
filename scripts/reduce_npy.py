"""Spatially decimate a large .npy array for fast prototyping.

Source: source-temp/reduce_npy.py

Two data-type modes:
  sentinel        shape (T, H, W, B) → data[:, ::d, ::d, ...]
  representations shape (H, W, C)    → data[::d, ::d, ...]

Uses mmap_mode='r' to avoid loading the full array into memory.

Usage
-----
python scripts/reduce_npy.py \\
  --input representations.npy \\
  --output representations_dec10.npy \\
  --decimation-factor 10 \\
  --data-type representations
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Spatially decimate a .npy array.")
    p.add_argument("--input", required=True, help="Path to input .npy file.")
    p.add_argument("--output", required=True, help="Path for output decimated .npy.")
    p.add_argument("--decimation-factor", type=int, default=10,
                   help="Spatial decimation factor (default: 10).")
    p.add_argument("--data-type", required=True, choices=["sentinel", "representations"],
                   help="'sentinel' for (T,H,W,B) or 'representations' for (H,W,C).")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    d = args.decimation_factor
    # mmap_mode='r' opens the file as a memory-mapped read-only array.
    # Only the sliced rows/columns are physically read into RAM, avoiding
    # loading the full multi-GB array — essential for large S2/S1 datasets.
    logging.info(f"Loading {args.input} with mmap_mode='r'")
    data = np.load(args.input, mmap_mode='r')
    logging.info(f"Original shape: {data.shape}")

    if args.data_type == "sentinel":
        # Shape (T, H, W, B): keep all time steps (axis 0) and bands (axis 3),
        # but take every d-th pixel along H (axis 1) and W (axis 2).
        decimated = data[:, ::d, ::d, ...]
    else:
        # Shape (H, W, C): take every d-th pixel along H (axis 0) and W (axis 1).
        decimated = data[::d, ::d, ...]

    logging.info(f"Decimated shape: {decimated.shape}")
    # np.save forces materialisation of the memmap slice into a new .npy file
    np.save(args.output, decimated)
    logging.info(f"Saved decimated data to {args.output}")


if __name__ == "__main__":
    main()
