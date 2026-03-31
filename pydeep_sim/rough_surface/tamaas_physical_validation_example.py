from pathlib import Path

import numpy as np

experiment_name = "tamaas_physical_validation_example"
output_dir = "./"
experiment_dir = Path(output_dir) / experiment_name
experiment_dir.mkdir(parents=True, exist_ok=True)

from pydeep_sim.rough_surface.tamaas_queens_driver import Tamaas
from pydeep_sim.rough_surface.rough_surface_parameters import (
    TAMAAS_ROUGH_SURFACE_PARAMETERS,
)
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area_load_path,
)

num_grid_points_per_side = 512
num_pressure_steps = 50

master_seed = 260331
master_rng = np.random.default_rng(master_seed)

num_samples = 1
hurst_values = np.random.uniform(0.6, 0.8, num_samples)
q1_values = np.random.choice(np.array([1, 4, 16]), size=num_samples)
q2_values = np.random.choice(np.array([32, 64, 128]), size=num_samples)
random_seeds = master_rng.integers(1, 2**31, size=num_samples, dtype=np.int32)

points = {
    "hurst": np.ravel(hurst_values),
    "q1": np.ravel(q1_values),
    "q2": np.ravel(q2_values),
    "random_seed": random_seeds,
}

sample = np.ravel([value for value in points.values()])

if __name__ == "__main__":

    print(points)
    print(sample)

    # Setup iterator
    tamaas_driver = Tamaas(
        parameters=TAMAAS_ROUGH_SURFACE_PARAMETERS,
        input_templates="",
        jobscript_template="",
        executable=None,
        lateral_length=1.0,
        plot_surface=True,
        num_grid_points_per_side=num_grid_points_per_side,
        target_pressure=0.40,
        num_pressure_steps=num_pressure_steps,
        solver_tolerance=1e-09,
        periodic=False,
        scale_surface=True,
        solve_contact_problem=True,
        random_load_steps=False,
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_load_path,
    )

    result, _ = tamaas_driver.run(
        sample,
        job_id=0,
        num_procs=1,
        experiment_name=experiment_name,
        experiment_dir=experiment_dir,
    )
    print(result)
