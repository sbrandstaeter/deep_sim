from queens.global_settings import GlobalSettings
from queens.iterators import Grid
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.mirco_queens_driver import MIRCO_DRIVER
from pydeep_sim.rough_surface.rough_surface_parameters import ROUGH_SURFACE_PARAMETERS


experiment_name = "queens_rough_surface_grid"
output_dir = "./"

global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

grid_design = {
    "hurst": {"num_grid_points": 3, "axis_type": "lin", "data_type": "FLOAT"},
    "far_field_displacement": {
        "num_grid_points": 3,
        "axis_type": "lin",
        "data_type": "FLOAT",
    },
}

if __name__ == "__main__":
    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=1,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=MIRCO_DRIVER)
    iterator = Grid(
        grid_design=grid_design,
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
