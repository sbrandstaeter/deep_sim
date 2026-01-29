import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from queens.distributions import Uniform
from queens.parameters import Parameters
from queens.global_settings import GlobalSettings

experiment_name = "tamaas_points_run_times_load_paths_1"
output_dir = "./"
global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

from queens.iterators import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.tamaas_queens_driver import TamaasRunTime
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area_load_path,
)

data = np.load("tamaas_compute_eff_area_over_pressure_for_run_time.npz")

q1 = data["q1"]
q2 = data["q2"]
hurst = data["hurst"]
seed = data["seed"]
surface = data["surface"]
A_raw = data["A_raw"]
A_cor = data["A_cor"]
loads = data["loads"]
rms_slope = data["rms_slope"]
Dmin = data["Dmin"]
Dmean = data["Dmean"]
Dmax = data["Dmax"]
run_time = data["run_time"]

# Build evaluation grid in A_cor space
area_range = np.arange(0.1, 1.0 + 1e-12, 0.1)
area_range[-1] = 0.998
# Linear interpolation; outside range -> NaN (change if you prefer clamping)
pressure = np.interp(area_range, A_cor, loads, left=np.nan, right=np.nan)


# plt.figure()
# plt.plot(loads, A_cor)
# plt.scatter(pressure, A_grid)
# plt.xlabel("Load")
# plt.ylabel("A_cor")
# plt.title("Corrected Contact Area vs Load")
# plt.tight_layout()
# plt.show()

num_pressure_steps = np.array([1, 2, 4, 8, 16, 32, 64])
pressure_grid, num_pressure_steps_grid = np.meshgrid(pressure, num_pressure_steps)
area_grid, num_pressure_steps_grid_2 = np.meshgrid(area_range, num_pressure_steps)

num_samples = pressure_grid.size

points = {
    "pressure_target": np.ravel(pressure_grid),
    "num_pressure_steps": np.ravel(num_pressure_steps_grid),
}

np.savez(
    Path(output_dir) / f"{experiment_name}_samples.npz",
    eff_area=area_grid.ravel(),
    **points,
)

print(points)
if __name__ == "__main__":

    print(points)
    tamaas_run_time_parameters = Parameters(
        num_pressure_steps=Uniform(lower_bound=1, upper_bound=1024),
        pressure_target=Uniform(lower_bound=0, upper_bound=10),
    )
    # Setup iterator
    tamaas_driver = TamaasRunTime(
        parameters=tamaas_run_time_parameters,
        input_templates="",
        jobscript_template="",
        executable=None,
        lateral_length=1.0,
        plot_surface=True,
        num_grid_points_per_side=512,
        solver_tolerance=1e-09,
        periodic=False,
        scale_surface=True,
        solve_contact_problem=True,
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_load_path,
        q1=1,
        q2=32,
        hurst=0.7,
        random_seed=seed,
    )
    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=np.min([60, num_samples]),
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=tamaas_driver)
    iterator = Points(
        points=points,
        result_description={"write_results": True},
        model=model,
        parameters=tamaas_run_time_parameters,
        global_settings=global_settings,
    )

    with global_settings:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))
