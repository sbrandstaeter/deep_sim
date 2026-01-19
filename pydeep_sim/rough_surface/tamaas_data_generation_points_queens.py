import numpy as np

from queens.global_settings import GlobalSettings

experiment_name = "tamaas_points_periodic_1"
output_dir = "./"
global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

from queens.iterators import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.tamaas_queens_driver import TAMAAS_DRIVER
from pydeep_sim.rough_surface.rough_surface_parameters import (
    TAMAAS_ROUGH_SURFACE_PARAMETERS,
)

master_seed = 931990
master_rng = np.random.default_rng(master_seed)

# hurst_values = np.array([0.5, 0.6, 0.7, 0.8])
hurst_values = np.linspace(0.6, 0.8, 50)
q1_values = np.array([1, 4, 16])
q2_values = np.array([32, 64, 128])


hurst_grid, q1_grid, q2_grid = np.meshgrid(hurst_values, q1_values, q2_values)

num_samples = hurst_grid.size
random_seeds = master_rng.integers(0, 2**64, size=num_samples, dtype=np.uint64)

random_seeds = master_rng.integers(1, 2**31, size=num_samples, dtype=np.int32)

points = {
    "hurst": np.ravel(hurst_grid),
    "q1": np.ravel(q1_grid),
    "q2": np.ravel(q2_grid),
    "random_seed": random_seeds,
}


if __name__ == "__main__":

    print(points)

    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=60,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=TAMAAS_DRIVER)
    iterator = Points(
        points=points,
        result_description={"write_results": True},
        model=model,
        parameters=TAMAAS_ROUGH_SURFACE_PARAMETERS,
        global_settings=global_settings,
    )

    with global_settings:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))
