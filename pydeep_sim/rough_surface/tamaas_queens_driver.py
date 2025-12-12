from itertools import chain
from pathlib import Path

import numpy as np
from queens.drivers import Jobscript
from queens.drivers.jobscript import JobOptions
from queens.utils.injector import inject, inject_in_template
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata

from pydeep_sim.rough_surface.rough_surface import (
    plot_surface,
    random_postprocess,
)
from pydeep_sim.rough_surface.patches_generation import (
    plot_cumulative_distribution,
    plot_probability_density,
)
from pydeep_sim.rough_surface.tamaas_load_path import (
    generate_surface_and_solve_pressure_driven_eff_area,
)
from pydeep_sim.rough_surface.rough_surface_parameters import (
    TAMAAS_ROUGH_SURFACE_PARAMETERS,
)


class Tamaas(Jobscript):

    @log_init_args
    def __init__(
        self,
        *args,
        lateral_length=1.0,
        num_pressure_steps=10,
        plot_surface=False,
        num_grid_points_per_side=512,
        target_pressure=0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lateral_length = lateral_length
        self.num_pressure_steps = num_pressure_steps
        self.plot_surface = plot_surface
        self.num_grid_points_per_side = num_grid_points_per_side
        self.target_pressure = target_pressure

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample (dict): Dict containing sample.
            job_id (int): Job ID.
            num_procs (int): Number of processors.
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): Name of QUEENS experiment.

        Returns:
            Result and potentially the gradient.
        """
        job_dir, output_dir, output_file, input_files, log_file = self._manage_paths(
            job_id, experiment_dir
        )

        sample_dict = self.parameters.sample_as_dict(sample)

        metadata = SimulationMetadata(
            job_id=job_id, inputs=sample_dict, job_dir=job_dir
        )

        with metadata.time_code("create_surface_and_solve_contact_problem"):
            final_surface_name = "topology_RMD_aggregated.dat"
            surface_path = job_dir / final_surface_name

            (
                rough_surface,
                A_raw,
                A_cor,
                loads,
                rms_slope,
                Dmin,
                Dmean,
                Dmax,
                run_times,
            ) = generate_surface_and_solve_pressure_driven_eff_area(
                hurst=sample_dict["hurst"],
                q1=int(sample_dict["q1"]),
                q2=int(sample_dict["q2"]),
                n=self.num_grid_points_per_side,
                L=self.lateral_length,
                random_seed=int(sample_dict["random_seed"]),
                p_target=self.target_pressure,
                num_load_steps=self.num_pressure_steps,
            )
            np.savetxt(
                surface_path,
                rough_surface,
                fmt="%25.17e",
                delimiter=";",
            )

            sample_dict["surface_path"] = surface_path
            sample_dict["lateral_length"] = self.lateral_length

            if self.plot_surface:
                plot_surface(
                    rough_surface=rough_surface,
                    lateral_length=self.lateral_length,
                    output_dir=output_dir,
                )
                plot_probability_density(
                    rough_surface,
                    num_patches=sample_dict["q1"],
                    path_to_figure=output_dir / f"pdf.png",
                )
                plot_cumulative_distribution(
                    rough_surface,
                    num_patches=sample_dict["q1"],
                    path_to_figure=output_dir / f"cdf.png",
                    probability=np.max(A_cor),
                )

            statistical_properties = random_postprocess(
                rough_surface=rough_surface, lateral_length=self.lateral_length
            )
            # overwrite with original rms_slope prior to scaling the surface
            statistical_properties["rms_slope_peaks"] = rms_slope

            statistical_properties_results = np.array(
                list(statistical_properties.values())
            )

            statistical_properties_results = np.repeat(
                np.atleast_2d(statistical_properties_results),
                self.num_pressure_steps,
                axis=0,
            )

            Dmean = np.array(Dmean)
            Dmin = np.array(Dmin)
            Dmax = np.array(Dmax)
            run_times = np.array(run_times)

            overall_result = np.hstack(
                [
                    statistical_properties_results,
                    loads.reshape(-1, 1),
                    A_cor.reshape(-1, 1),
                    Dmean.reshape(-1, 1),
                    Dmin.reshape(-1, 1),
                    Dmax.reshape(-1, 1),
                    run_times.reshape(-1, 1),
                ]
            ).flatten()
            np.save(output_dir / "overall_result.npy", overall_result)
            gradient = None
            metadata.outputs = overall_result, gradient

        return overall_result, gradient


# Setup iterator
TAMAAS_DRIVER = Tamaas(
    parameters=TAMAAS_ROUGH_SURFACE_PARAMETERS,
    input_templates="",
    jobscript_template="",
    executable=None,
    lateral_length=1.0,
    plot_surface=True,
    num_grid_points_per_side=512,
    target_pressure=0.1,
    num_pressure_steps=50,
)
