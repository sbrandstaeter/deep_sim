import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from pydeep_sim.rough_surface.rough_surface_parameters import (
    TAMAAS_ROUGH_SURFACE_PARAMETERS,
    TAMAAS_ROUGH_SURFACE_PARAMETERS_PRESSURE,
)
from pydeep_sim.rough_surface.tamaas_queens_driver import scale_factor, Tamaas
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps,
)

experiment_name = "tamaas_test_generation_test_data"
experiment_dir = pathlib.Path(".") / experiment_name

max_workers = 60  # or set to os.cpu_count()

num_grid_points_per_side = 512
num_pressure_steps = 1

master_seed = 260312
master_rng = np.random.default_rng(master_seed)

num_samples = 4500
hurst_values = np.sort(np.random.uniform(0.6, 0.8, num_samples))
q1_values = np.random.choice(np.array([1, 4, 16]), size=num_samples)
q2_values = np.random.choice(np.array([32, 64, 128]), size=num_samples)
random_seeds = master_rng.integers(1, 2**31, size=num_samples, dtype=np.int32)
target_pressure = np.random.uniform(0.0, 0.4, num_samples)

points = {
    "hurst": np.ravel(hurst_values),
    "q1": np.ravel(q1_values),
    "q2": np.ravel(q2_values),
    "random_seed": random_seeds,
    "target_pressure": np.ravel(target_pressure),
}


def run_single_sample(i, hurst, q1, q2, random_seed, target_pressure):
    start_time = time.time()
    # Build driver inside the worker process
    tamaas_driver = Tamaas(
        parameters=TAMAAS_ROUGH_SURFACE_PARAMETERS_PRESSURE,
        input_templates="",
        jobscript_template="",
        executable=None,
        lateral_length=1.0,
        plot_surface=True,
        num_grid_points_per_side=num_grid_points_per_side,
        target_pressure=target_pressure,
        num_pressure_steps=num_pressure_steps,
        solver_tolerance=1e-09,
        periodic=False,
        scale_surface=True,
        solve_contact_problem=True,
        random_load_steps=False,
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps,
    )

    overall_result, gradient = tamaas_driver.run(
        sample=np.array([hurst, q1, q2, random_seed, target_pressure]),
        job_id=i,
        num_procs=1,
        experiment_dir=experiment_dir,
        experiment_name=experiment_name,
    )

    elapsed = time.time() - start_time

    return {"i": i, "result": overall_result}


if __name__ == "__main__":
    results = [None] * num_samples

    print(f"Start computing {num_samples} samples on {max_workers} workers.")
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_sample,
                i,
                float(hurst_values[i]),
                int(q1_values[i]),
                int(q2_values[i]),
                int(random_seeds[i]),
                float(target_pressure[i]),
            )
            for i in range(num_samples)
        ]

        for future in tqdm(
            as_completed(futures),
            total=num_samples,
            desc="Solving rough surface contact",
        ):
            result = future.result()
            i = result["i"]

            results[i] = result["result"]

    print(f"Finished entire computation in {time.time() - start_time}s.")

    np.savez(
        "test_generation_test_data.npz",
        results=np.array(results),
    )
