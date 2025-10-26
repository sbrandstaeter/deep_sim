from itertools import chain
from pathlib import Path

import numpy as np
from queens.drivers import Jobscript
from queens.drivers.jobscript import JobOptions
from queens.utils.injector import inject, inject_in_template
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata

from pydeep_sim.rough_surface.rough_surface import (
    RoughSurface,
    plot_surface,
    random_postprocess,
)
from pydeep_sim.rough_surface.patches_generation import (
    patches_generation,
    plot_cumulative_distribution,
    plot_probability_density,
)
from pydeep_sim.rough_surface.rough_surface_parameters import ROUGH_SURFACE_PARAMETERS
from pydeep_sim.rough_surface.mirco_effective_contact_area_data_processor import (
    MIRCO_EFFECTIVE_CONTACT_AREA_DATAPROCESSOR,
)

JOBSCRIPT_LOCAL_HEADER = """
#!/bin/bash
# Setup shell environment and start from home dir
"""

JOBSCRIPT_CLUSTER_HEADER = """
#!/bin/bash
# Setup shell environment and start from home dir

source /home/cluster_tools/user/load_four_c_environment.sh

module list
"""

JOBSCRIPT_TEMPLATE_CORE = """
MIRCO_EXE={{ executable }}

MIRCO_INPUT_FILE={{ mirco_input_file }}

OUTPUT_DIR={{ output_dir }}


$MIRCO_EXE $MIRCO_INPUT_FILE 

wait

echo  
echo Time is `date` 
echo Directory is `pwd` 
echo 'Computations are done. About to finish the job.' 

echo
echo "Job finished with exit code $? at: `date`"
"""

JOBSCRIPT_TEMPLATE = JOBSCRIPT_LOCAL_HEADER + JOBSCRIPT_TEMPLATE_CORE

JOBSCRIPT_CLUSTER_TEMPLATE = JOBSCRIPT_CLUSTER_HEADER + JOBSCRIPT_TEMPLATE_CORE


class MircoJobscript(Jobscript):

    @log_init_args
    def __init__(
        self,
        *args,
        rmd_resolution=7,
        lateral_length=1.0,
        initial_topology_std_deviation=1.0,
        max_effective_contact_area=0.1,
        num_far_field_displacements=1,
        plot_surface=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rmd_resolution = rmd_resolution
        self.lateral_length = lateral_length
        self.initial_topology_std_dev = initial_topology_std_deviation
        self.max_effective_contact_area = max_effective_contact_area
        self.num_far_field_displacements = num_far_field_displacements
        self.plot_surface = plot_surface

    def prepare_input_file(self, sample_dict, experiment_dir, input_file):
        """prepare and parse data to input files.

        args:
            sample_dict (dict): dict containing sample
            experiment_dir (path): path to queens experiment directory.
            input_file (path): path of the input file
        """
        # we only expect one input template here
        input_template_name, input_template_path = list(self.input_templates.items())[0]
        inject(
            sample_dict,
            experiment_dir / input_template_path.name,
            input_file,
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

        N = 2**self.rmd_resolution

        if sample_dict.get("num_patches") is None:
            num_patches = 1
            sample_dict["num_patches"] = num_patches
        else:
            num_patches = int(sample_dict["num_patches"])

        metadata = SimulationMetadata(
            job_id=job_id, inputs=sample_dict, job_dir=job_dir
        )

        with metadata.time_code("create_and_analyse_surface"):
            final_surface_name = "topology_RMD_aggregated.dat"
            surface_path = job_dir / final_surface_name
            rough_surface = patches_generation(
                H=sample_dict["hurst"],
                n_iter=num_patches,
                surf_id=job_id,
                path_to_patches=job_dir / "patches",
                path_to_surface=surface_path,
                file_tail="RMD_" + str(job_id),
                N=N,
                l=self.lateral_length,
                std0=self.initial_topology_std_dev,
            )

            sample_dict["surface_path"] = surface_path
            sample_dict["lateral_length"] = self.lateral_length

            if sample_dict.get("far_field_displacement") is None:
                max_far_field_displacement = np.max(rough_surface) - np.quantile(
                    np.ravel(rough_surface), 1 - self.max_effective_contact_area
                )

                far_field_displacements = np.linspace(
                    0.0,
                    max_far_field_displacement,
                    num=self.num_far_field_displacements + 1,
                )
                # discard far_field_discplacement = 0.0
                far_field_displacements = far_field_displacements[1:]
            else:
                far_field_displacements = np.array(
                    [sample_dict["far_field_displacement"]]
                )

            if self.plot_surface:
                plot_surface(
                    rough_surface=rough_surface,
                    lateral_length=self.lateral_length,
                    output_dir=output_dir,
                )
                plot_probability_density(
                    rough_surface,
                    num_patches=num_patches,
                    path_to_figure=output_dir / f"pdf.png",
                )
                plot_cumulative_distribution(
                    rough_surface,
                    num_patches=num_patches,
                    path_to_figure=output_dir / f"cdf.png",
                    probability=self.max_effective_contact_area,
                )

            statistical_properties = random_postprocess(
                rough_surface=rough_surface, lateral_length=self.lateral_length
            )

            statistical_properties_results = np.array(
                list(statistical_properties.values())
            )

        mirco_results = []
        run_times = []
        for i, far_field_displacement in enumerate(far_field_displacements):

            sample_dict["far_field_displacement"] = far_field_displacement
            sub_output_dir = output_dir / str(i)
            sub_output_dir.mkdir(exist_ok=True, parents=True)
            input_template_name, input_template_path = list(
                self.input_templates.items()
            )[0]
            input_file_str = f"mirco_input_{i}" + input_template_path.suffix
            input_file = job_dir / input_file_str
            input_files[input_template_name] = input_file

            job_script_file_name, job_script_file_suffix = (
                self.jobscript_file_name.split(".")
            )
            jobscript_file = job_dir / (
                job_script_file_name + f"_{i}." + job_script_file_suffix
            )
            log_file = sub_output_dir / "output.log"

            with metadata.time_code(f"prepare_input_files_{i}"):
                job_options = JobOptions(
                    job_dir=job_dir,
                    output_dir=sub_output_dir,
                    output_file=output_file,
                    job_id=job_id,
                    num_procs=num_procs,
                    experiment_dir=experiment_dir,
                    experiment_name=experiment_name,
                    input_files=input_files,
                )

                # Create the input files
                self.prepare_input_file(
                    job_options.add_data_and_to_dict(sample_dict),
                    experiment_dir,
                    input_file,
                )

                # Create jobscript
                inject_in_template(
                    job_options.add_data_and_to_dict(self.jobscript_options),
                    self.jobscript_template,
                    str(jobscript_file),
                )

            mirco_computation_section_name = f"run_jobscript_{i}"
            with metadata.time_code(mirco_computation_section_name):
                execute_cmd = f"bash {jobscript_file} >{log_file} 2>&1"
                self._run_executable(job_id, execute_cmd)
            run_times.append(metadata.times[mirco_computation_section_name]["time"])

            with metadata.time_code(f"data_processing_{i}"):
                mirco_result, gradient = self._get_results(sub_output_dir)
                mirco_results.append(mirco_result)

        with metadata.time_code("finalize_output"):

            try:
                ravel_mirco_results = np.ravel(mirco_results)
            except Exception as e:
                print("An error occurred while raveling 'mirco_results':", e)
                print("Type of error:", type(e).__name__)
                print("Problematic input:", mirco_results)
                print(f"job_id: {job_id}, sub_job_id: {i}")
                raise  # re-raises the same exception
            overall_result = np.concatenate(
                [
                    statistical_properties_results,
                    far_field_displacements,
                    ravel_mirco_results,
                    run_times,
                ]
            )
            metadata.outputs = overall_result, gradient

        return overall_result, gradient


# Setup iterator
MIRCO_DRIVER = MircoJobscript(
    parameters=ROUGH_SURFACE_PARAMETERS,
    input_templates={"mirco_input_file": "./mirco_input_template.yml"},
    jobscript_template=JOBSCRIPT_CLUSTER_TEMPLATE,
    executable="/home/a11bsebr/codespace/mirco/build/mirco",
    files_to_copy=None,
    data_processor=MIRCO_EFFECTIVE_CONTACT_AREA_DATAPROCESSOR,
    gradient_data_processor=None,
    jobscript_file_name="jobscript.sh",
    extra_options=None,
    raise_error_on_jobscript_failure=True,
    initial_topology_std_deviation=90.0,
    lateral_length=1000.0,
    max_effective_contact_area=0.1,
    num_far_field_displacements=10,
    plot_surface=True,
)
