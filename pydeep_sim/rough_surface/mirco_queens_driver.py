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
        plot_surface=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rmd_resolution = rmd_resolution
        self.lateral_length = lateral_length
        self.initial_topology_std_dev = initial_topology_std_deviation
        self.max_effective_contact_area = max_effective_contact_area
        self.plot_surface = plot_surface

    def prepare_input_files(self, sample_dict, experiment_dir, input_files):
        """prepare and parse data to input files.

        args:
            sample_dict (dict): dict containing sample
            experiment_dir (path): path to queens experiment directory.
            input_files (dict): dict with name and path of the input file(s)
        """
        for input_template_name, input_template_path in self.input_templates.items():
            inject(
                sample_dict,
                experiment_dir / input_template_path.name,
                input_files[input_template_name],
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
                max_far_field_displacement = np.quantile(
                    np.ravel(rough_surface), self.max_effective_contact_area
                )
                sample_dict["far_field_displacement"] = max_far_field_displacement

            np.testing.assert_allclose(
                rough_surface, np.loadtxt(surface_path, delimiter=";")
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

        with metadata.time_code("prepare_input_files"):
            job_options = JobOptions(
                job_dir=job_dir,
                output_dir=output_dir,
                output_file=output_file,
                job_id=job_id,
                num_procs=num_procs,
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                input_files=input_files,
            )

            # Create the input files
            self.prepare_input_files(
                job_options.add_data_and_to_dict(sample_dict),
                experiment_dir,
                input_files,
            )

            jobscript_file = job_dir / self.jobscript_file_name

            # Create jobscript
            inject_in_template(
                job_options.add_data_and_to_dict(self.jobscript_options),
                self.jobscript_template,
                str(jobscript_file),
            )

        with metadata.time_code("run_jobscript"):
            execute_cmd = f"bash {jobscript_file} >{log_file} 2>&1"
            self._run_executable(job_id, execute_cmd)

        with metadata.time_code("data_processing"):
            mirco_result, gradient = self._get_results(output_dir)
            if mirco_result is not None:
                overall_result = np.concatenate(
                    [statistical_properties_results, mirco_result]
                )
            else:
                overall_result = statistical_properties_results
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
    max_effective_contact_area=0.2,
    plot_surface=True,
)
