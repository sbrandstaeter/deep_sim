from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    experiment_name = "tamaas_points_run_times_load_paths_1"
    output_dir = Path("./")
    with np.load(output_dir / f"{experiment_name}_database.npz") as data:
        wall_clock_times = data["wall_clock_times"]
        pressure_target = data["pressure_target"]
        num_pressure_steps = data["num_pressure_steps"]
        eff_area = data["eff_area"]

    # example names
    x = num_pressure_steps
    y = eff_area
    c = wall_clock_times

    colormap = "plasma"
    plt.figure()
    sc = plt.scatter(x, y, c=c, cmap=colormap)
    plt.colorbar(sc, label="Wall clock time")

    plt.xscale("log")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    X = num_pressure_steps.reshape(7, 10)
    Y = eff_area.reshape(7, 10)
    C = wall_clock_times.reshape(7, 10)
    plt.figure()
    pc = plt.pcolormesh(X, Y, C, shading="nearest", cmap=colormap)
    plt.colorbar(pc, label="Wall clock time")

    plt.xscale("log")  # log x still works
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    plt.figure()

    cf = plt.tricontourf(
        x,
        y,
        c,
        levels=10,  # increase for smoother gradients
        cmap=colormap,
    )

    plt.colorbar(cf, label="Wall clock time")
    plt.xscale("log")  # keep log x-axis
    plt.xlabel("Number of pressure steps")
    plt.ylabel("Effective contact area")
    plt.tight_layout()
    plt.show()
