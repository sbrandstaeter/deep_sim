from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from queens.utils.io import load_result

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def read_results_per_subdir(
    root: Path, filename: str = "overall_result.npy"
) -> dict[str, dict[int, np.ndarray]]:
    """
    For each immediate subdirectory of `root`, look for `filename`.
    If found, load as a ';' delimited 2D array and compute std (over all elements).
    Returns: {subdir_name: std_value}
    """

    if not root.exists() or not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    # column order MUST match how the file was saved
    columns = ["loads", "A_cor", "Dmean", "Dmin", "Dmax", "run_times"]

    # initialize results dict automatically
    results: dict[str, dict[int, np.ndarray]] = {name: {} for name in columns}

    for subdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        data_path = subdir / "output" / filename
        if not data_path.is_file():
            continue

        try:
            overall_result = np.load(data_path)
            N = overall_result.size // 6

            overall_result = overall_result.reshape(N, 6)
            key = int(subdir.name)

            # split columns generically
            for i, name in enumerate(columns):
                results[name][key] = overall_result[:, i]

        except Exception as e:
            print(f"[WARN] Failed to process {data_path}: {e}")

    return results


if __name__ == "__main__":

    experiment_name = "tamaas_points_run_times_load_paths_1"
    output_dir = Path("./")
    root = Path().home() / "queens-experiments" / experiment_name

    results = read_results_per_subdir(root=root, filename="overall_result.npy")

    np.save(output_dir / f"{experiment_name}_results.npz", results)
    samples = np.load(output_dir / f"{experiment_name}_samples.npz")

    run_times_dict = results["run_times"]
    wall_clock_times_dict = {}
    for id, run_times in run_times_dict.items():
        wall_clock_times_dict[id] = np.sum(run_times)

    ids = np.array(list(wall_clock_times_dict.keys()), dtype=int)
    stds = np.array(
        [wall_clock_times_dict[k] for k in wall_clock_times_dict.keys()], dtype=float
    )
    wall_clock_times = np.zeros(len(ids))
    wall_clock_times[ids] = stds

    np.savez(
        output_dir / f"{experiment_name}_database.npz",
        wall_clock_times=wall_clock_times,
        pressure_target=samples["pressure_target"],
        num_pressure_steps=samples["num_pressure_steps"],
        eff_area=samples["eff_area"],
    )
