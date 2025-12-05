import numpy as np

from queens.global_settings import GlobalSettings

experiment_name = "rough_surface_points_16"
output_dir = "./"
global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

from queens.iterators import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.mirco_queens_driver import MIRCO_DRIVER
from pydeep_sim.rough_surface.rough_surface_parameters import ROUGH_SURFACE_PARAMETERS


# hurst_values = np.array([0.5, 0.6, 0.7, 0.8])
hurst_values = np.linspace(0.6, 0.8, 50)
num_patches_values = np.array([1, 4, 16, 64])

hurst_grid, num_patches_grid = np.meshgrid(hurst_values, num_patches_values)

points = {"hurst": np.ravel(hurst_grid), "num_patches": np.ravel(num_patches_grid)}


if __name__ == "__main__":

    print(points)

    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=20,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=MIRCO_DRIVER)
    iterator = Points(
        points=points,
        result_description={"write_results": True},
        model=model,
        parameters=ROUGH_SURFACE_PARAMETERS,
        global_settings=global_settings,
    )

    with global_settings:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))
