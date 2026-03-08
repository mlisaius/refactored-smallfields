"""Remap 17-class crop labels to the 7-class small-fields scheme.

Source: source-temp/remap_cropcodes_forsmallfields.py

Class mapping (17-class → 7-class):
  1  → 1  (Legume)
  2  → 2  (Soy)
  4  → 3  (Winter grain)
  5  → 4  (Maize)
  7  → 5  (Potato)
  9  → 6  (Squash)
  3,6,8,10,11,14 → 7  (Other crop)
  12,13,15,16,17 → 0  (Discard)
  all others     → 0  (Discard)

Usage
-----
python scripts/remap_cropcodes.py \\
  --input fieldtype_17classes.npy \\
  --output fieldtype_7classes_maddysmallfields.npy
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="Remap 17-class labels to 7-class scheme.")
    p.add_argument("--input", required=True, help="Path to 17-class .npy label array.")
    p.add_argument("--output", required=True, help="Path for 7-class output .npy.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(f"Loading labels from {args.input}")
    arr = np.load(args.input)
    logging.info(f"Input shape: {arr.shape}, unique values: {np.unique(arr)}")

    # Initialise with zeros (discard) — any 17-class code not explicitly mapped
    # below defaults to 0.  This matches the source exactly.
    remapped = np.zeros_like(arr)

    # Apply per-class mappings using vectorised np.where.
    # Each call replaces the output value at positions where arr matches.
    # Order matters: the last matching np.where wins, so put more specific
    # mappings last (though in practice each source class maps to exactly one target).
    remapped = np.where(arr == 1, 1, remapped)                             # Legume
    remapped = np.where(arr == 2, 2, remapped)                             # Soy
    remapped = np.where(arr == 4, 3, remapped)                             # Winter grain
    remapped = np.where(arr == 5, 4, remapped)                             # Maize
    remapped = np.where(arr == 7, 5, remapped)                             # Potato
    remapped = np.where(arr == 9, 6, remapped)                             # Squash
    remapped = np.where(np.isin(arr, [3, 6, 8, 10, 11, 14]), 7, remapped) # Other crops
    # Classes 12,13,15,16,17 → 0 (Discard); np.isin is faster than chained np.where
    remapped = np.where(np.isin(arr, [12, 13, 15, 16, 17]), 0, remapped)   # Discard

    logging.info(f"Output shape: {remapped.shape}, unique values: {np.unique(remapped)}")
    np.save(args.output, remapped)
    logging.info(f"Saved remapped array to {args.output}")


if __name__ == "__main__":
    main()
