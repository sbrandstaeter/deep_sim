import numpy as np

from queens.distributions import Uniform
from queens.global_settings import GlobalSettings
from queens.iterators import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers import Local
from queens.utils.io import load_result

from pydeep_sim.rough_surface.mirco_queens_driver import (
    MircoJobscript,
    JOBSCRIPT_TEMPLATE,
)

experiment_name = "queens_rough_surface"
output_dir = "./"

global_settings = GlobalSettings(experiment_name=experiment_name, output_dir=output_dir)

hurst = Uniform(lower_bound=0.50, upper_bound=0.80)
far_field_displacement = Uniform(lower_bound=5.0, upper_bound=45.0)

parameters = Parameters(hurst=hurst, far_field_displacement=far_field_displacement)

# Setup iterator
driver = MircoJobscript(
    parameters=parameters,
    input_templates={"mirco_input_file": "./mirco_input_template.yml"},
    jobscript_template=JOBSCRIPT_TEMPLATE,
    executable="path-to-mirco",
    files_to_copy=None,
    data_processor=None,
    gradient_data_processor=None,
    jobscript_file_name="jobscript.sh",
    extra_options=None,
    raise_error_on_jobscript_failure=True,
    initial_topology_std_deviation=20.0,
    lateral_length=1000.0,
    plot_surface=True,
)

num_points = 10
hurst_values = np.array([0.55] * num_points)
far_field_displacement = np.array([5.0] * num_points)
points = {"hurst": hurst_values, "far_field_displacement": far_field_displacement}

if __name__ == "__main__":
    # scheduler = Pool(experiment_name=global_settings.experiment_name)
    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_jobs=1,
        num_procs=1,
        restart_workers=False,
        verbose=True,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Points(
        points=points,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    with global_settings:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))
