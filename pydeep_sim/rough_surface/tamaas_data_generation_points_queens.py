import numpy as np

from queens.global_settings import GlobalSettings

# experiment_name = "tamaas_points_surfaces_scaled"
experiment_name = "tamaas_points_nonperiodic_3_indiv_load_steps"
# experiment_name = "tamaas_points_nonperiodic_3"
# experiment_name = "tamaas_points_periodic_2"
experiment_name = "tamaas_points_nonperiodic_4_indiv_load_steps"
output_dir = "./"
global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

from queens.iterators import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.tamaas_queens_driver import Tamaas
from pydeep_sim.rough_surface.rough_surface_parameters import (
    TAMAAS_ROUGH_SURFACE_PARAMETERS,
)
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area_load_path,
    generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps,
)

master_seed = 931990
# master_seed = 260124
num_grid_points_per_side = 512
num_pressure_steps = 25
master_rng = np.random.default_rng(master_seed)

# hurst_values = np.array([0.5, 0.6, 0.7, 0.8])
hurst_values = np.linspace(0.6, 0.8, 25)
q1_values = np.array([1, 4, 16])
q2_values = np.array([32, 64, 128])

hurst_grid, q1_grid, q2_grid = np.meshgrid(hurst_values, q1_values, q2_values)

num_samples = hurst_grid.size

random_seeds = master_rng.integers(1, 2**31, size=num_samples, dtype=np.int32)

points = {
    "hurst": np.ravel(hurst_grid),
    "q1": np.ravel(q1_grid),
    "q2": np.ravel(q2_grid),
    "random_seed": random_seeds,
}

if __name__ == "__main__":

    print(points)

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
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_individual_load_steps,
    )
    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=60,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=tamaas_driver)
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
