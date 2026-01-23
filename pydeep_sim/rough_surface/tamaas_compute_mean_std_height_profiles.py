#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def compute_std_per_subdir(
    root: Path, filename: str = "topology_RMD_aggregated.dat"
) -> dict[int, float]:
    """
    For each immediate subdirectory of `root`, look for `filename`.
    If found, load as a ';' delimited 2D array and compute std (over all elements).
    Returns: {subdir_name: std_value}
    """
    results: dict[int, float] = {}

    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        data_path = subdir / filename
        if not data_path.is_file():
            continue

        try:
            arr = np.loadtxt(data_path, delimiter=";", dtype=float)
            # Standard deviation over all entries in the matrix
            std_val = float(np.std(arr))
            results[int(subdir.name)] = std_val
        except Exception as e:
            print(f"[WARN] Failed to process {data_path}: {e}")

    return results


if __name__ == "__main__":
    experiment_name = "tamaas_points_nonperiodic_3"
    home = Path.home()
    root_path = home / Path(f"queens-experiments/{experiment_name}")
    filename = "topology_RMD_aggregated.dat"

    results = compute_std_per_subdir(root_path, filename)

    # Build aligned arrays (IDs + stds) if you want them as numpy arrays
    ids = np.array(list(results.keys()), dtype=int)
    stds = np.array([results[k] for k in results.keys()], dtype=float)

    stds_sorted = np.zeros(len(ids))
    stds_sorted[ids] = stds

    for key, value in results.items():
        np.testing.assert_array_equal(stds_sorted[key], value)

    mean_stds = np.mean(stds_sorted)

    np.save(f"{experiment_name}_surfaces_stds.npy", stds)
    np.save(f"{experiment_name}_surfaces_stds_mean.npy", mean_stds)

    print(results)
