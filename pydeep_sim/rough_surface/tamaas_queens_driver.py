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
    generate_surface_and_solve_pressure_driven_eff_area_load_path,
)

SCALE_FACTORS = [
    55.12229037844432,
    69.86908217092252,
    87.14220958058162,
    518.344463910421,
    680.1532036436737,
    872.7045726294955,
    3751.4321198525017,
    5944.4843376218405,
    8232.078271477685,
]

SCALE_FACTOR_ID_MAPPING = {
    1: {32: 0, 64: 1, 128: 2},
    4: {32: 3, 64: 4, 128: 5},
    16: {32: 6, 64: 7, 128: 8},
}


def scale_factor(q1, q2):
    return SCALE_FACTORS[SCALE_FACTOR_ID_MAPPING[q1][q2]]


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
        solver_tolerance=1e-09,
        periodic=False,
        scale_surface=False,
        solve_contact_problem=True,
        random_load_steps=False,
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_load_path,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Overwrite: force "no copy" while keeping the interface unchanged in the
        # Jobscript driver. Setting something like input_template="" is dangerous
        # because Path("") resolves to ".", which would cause rsync to copy the
        # current working directory. Instead we explicitly set the list of files
        # to copy to empty so that no rsync command is executed.
        self.files_to_copy = []

        self.lateral_length = lateral_length
        self.num_pressure_steps = num_pressure_steps
        self.plot_surface = plot_surface
        self.num_grid_points_per_side = num_grid_points_per_side
        self.target_pressure = target_pressure
        self.solver_tolerance = solver_tolerance
        self.periodic = periodic
        self.scale_surface = scale_surface
        self.solve_contact_problem = solve_contact_problem
        self.random_load_steps = random_load_steps
        self.generate_surface_and_solve_pressure_driven_eff_area = (
            generate_surface_and_solve_pressure_driven_eff_area
        )

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
            q1 = int(sample_dict["q1"])
            q2 = int(sample_dict["q2"])
            target_pressure = sample_dict.get("target_pressure", self.target_pressure)

            if self.scale_surface:
                scale_factor_surface = scale_factor(q1, q2)
            else:
                scale_factor_surface = 1.0

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
            ) = self.generate_surface_and_solve_pressure_driven_eff_area(
                hurst=sample_dict["hurst"],
                q1=q1,
                q2=q2,
                n=self.num_grid_points_per_side,
                L=self.lateral_length,
                random_seed=int(sample_dict["random_seed"]),
                p_target=target_pressure,
                num_load_steps=self.num_pressure_steps,
                solver_tolerance=self.solver_tolerance,
                periodic=self.periodic,
                scale_factor_surface=scale_factor_surface,
                solve_contact_problem=self.solve_contact_problem,
                random_load_steps=self.random_load_steps,
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
                    num_patches=q1,
                    path_to_figure=output_dir / f"pdf.png",
                )
                plot_cumulative_distribution(
                    rough_surface,
                    num_patches=q1,
                    path_to_figure=output_dir / f"cdf.png",
                    probability=np.max(A_cor),
                )

            statistical_properties = random_postprocess(
                rough_surface=rough_surface, lateral_length=self.lateral_length
            )
            # overwrite with original rms_slope prior to scaling the surface
            statistical_properties["rms_slope"] = rms_slope

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


class TamaasRunTime(Jobscript):

    @log_init_args
    def __init__(
        self,
        *args,
        lateral_length=1.0,
        plot_surface=False,
        num_grid_points_per_side=512,
        solver_tolerance=1e-09,
        periodic=False,
        scale_surface=False,
        solve_contact_problem=True,
        generate_surface_and_solve_pressure_driven_eff_area=generate_surface_and_solve_pressure_driven_eff_area_load_path,
        q1=1,
        q2=32,
        hurst=0.7,
        random_seed=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lateral_length = lateral_length
        self.plot_surface = plot_surface
        self.num_grid_points_per_side = num_grid_points_per_side
        self.solver_tolerance = solver_tolerance
        self.periodic = periodic
        self.scale_surface = scale_surface
        self.solve_contact_problem = solve_contact_problem
        self.generate_surface_and_solve_pressure_driven_eff_area = (
            generate_surface_and_solve_pressure_driven_eff_area
        )
        self.q1 = q1
        self.q2 = q2
        self.hurst = hurst
        self.random_seed = random_seed

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

            q1 = self.q1
            q2 = self.q2
            hurst = self.hurst
            if self.scale_surface:
                scale_factor_surface = scale_factor(q1, q2)
            else:
                scale_factor_surface = 1.0
            p_target = sample_dict["pressure_target"]
            num_pressure_steps = int(sample_dict["num_pressure_steps"])
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
            ) = self.generate_surface_and_solve_pressure_driven_eff_area(
                hurst=hurst,
                q1=q1,
                q2=q2,
                n=self.num_grid_points_per_side,
                L=self.lateral_length,
                random_seed=self.random_seed,
                p_target=p_target,
                num_load_steps=num_pressure_steps,
                solver_tolerance=self.solver_tolerance,
                periodic=self.periodic,
                scale_factor_surface=scale_factor_surface,
                solve_contact_problem=self.solve_contact_problem,
            )

            Dmean = np.array(Dmean)
            Dmin = np.array(Dmin)
            Dmax = np.array(Dmax)
            run_times = np.array(run_times)

            overall_result = np.hstack(
                [
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
