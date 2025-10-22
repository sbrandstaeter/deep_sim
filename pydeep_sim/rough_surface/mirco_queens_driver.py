from itertools import chain
from pathlib import Path

import numpy as np
from queens.drivers import Jobscript
from queens.drivers.jobscript import JobOptions
from queens.utils.injector import inject, inject_in_template
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata

from pydeep_sim.rough_surface.rough_surface import RoughSurface, plot_surface
from pydeep_sim.rough_surface.patches_generation import patches_generation
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
        plot_surface=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rmd_resolution = rmd_resolution
        self.lateral_length = lateral_length
        self.initial_topology_std_dev = initial_topology_std_deviation
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
        n_iter = 4
        final_surface_name = (
            "topology_RMD_aggregated_{0:02d}x{1:03d}_{2:04d}.dat".format(
                n_iter, int(N / np.sqrt(n_iter)), job_id
            )
        )
        surface_path = job_dir / final_surface_name
        rough_surface, topology = patches_generation(
            H=sample_dict["hurst"],
            n_iter=n_iter,
            surf_id=job_id,
            path_to_patches=job_dir / "patches",
            path_to_surface=surface_path,
            file_tail="RMD_" + str(job_id),
            N=N,
            l=self.lateral_length,
            std0=self.initial_topology_std_dev,
        )
        rough_surface -= np.min(rough_surface)

        sample_dict["surface_path"] = surface_path
        sample_dict["lateral_length"] = self.lateral_length

        np.testing.assert_allclose(
            rough_surface, np.loadtxt(surface_path, delimiter=";")
        )

        if self.plot_surface:
            plot_surface(
                rough_surface=rough_surface,
                lateral_length=self.lateral_length,
                output_dir=output_dir,
            )

        statistical_properties = {
            # peaks
            "mean_z_peaks": [],
            "rms_z_peaks": [],
            "ks_z_peaks": [],
            "sk_z_peaks": [],
            "mean_curv_peaks": [],
            "ks_curv_peaks": [],
            "sk_curv_peaks": [],
            "dn_peaks": [],
            "alfa_x": [],
            "alfa_y": [],
            # asperities
            "mean_z_asp": [],
            "rms_z_asp": [],
            "ks_z_asp": [],
            "sk_z_asp": [],
            "mean_curv_asp": [],
            "rms_curv_asp": [],
            "ks_curv_asp": [],
            "sk_curv_asp": [],
            "dns_asp": [],
            # surface statistics
            "z_mean": [],
            "z_max": [],
            "z_rms": [],
        }

        topology.random_postprocess(surface_path, statistical_properties)

        statistical_properties_results = np.array(
            list(chain.from_iterable(statistical_properties.values()))
        )

        metadata = SimulationMetadata(
            job_id=job_id, inputs=sample_dict, job_dir=job_dir
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
            result, gradient = self._get_results(output_dir)
            if result is not None:
                result = np.concatenate([statistical_properties_results, result])
            else:
                result = statistical_properties_results
            metadata.outputs = result, gradient

        return result, gradient


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
    initial_topology_std_deviation=20.0,
    lateral_length=1000.0,
    plot_surface=True,
)
