import numpy as np
from numpy.random import seed
from numpy import dtype, random as rnd
import subprocess
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from .iterator import Iterator
from ..rough_surface.rough_surface import RoughSurface
from ..sampling.sampling import Sampling

class RoughSurfaceBemRMDIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, num_simulations, result_description, driver, parameters, sampling, global_settings):
        super(RoughSurfaceBemRMDIterator, self).__init__(None, global_settings)
        self.num_simulations = num_simulations
        self.result_description = result_description
        self.driver = driver
        self.parameters = parameters
        self.sampling = sampling
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("global_settings", None)["experiment_name"])

        method_options = config["method"]["method_options"]
        num_simulations = method_options.get("num_simulations", None)
        result_description = method_options.get("result_description", None)
        
        driver = config.get("driver", None)
        parameters = config.get("parameters", None)
        sampling = config.get("sampling", None)
        global_settings = config.get("global_settings", None)

        return cls(num_simulations, result_description, driver, parameters, sampling, global_settings)

    def run_simulation(self):
        '''
        Run the BEM simulation 
        '''       
        # sampling methods require the ranges of the parameters
        # first get the parameters from the input file
        domain = self.get_parameters()

        # create sampling instance
        sample_model = Sampling(domain=list(domain.values()), 
                        n_samples=self.num_simulations, 
                        sampling_name=self.sampling["sampling_name"], 
                        sampling_options=self.sampling["sampling_options"])

        # generate the samples
        sampled_parameters = np.array(sample_model.generate_samples())
        
        # match the sampled parameters with the domain, 
        # so we are sure that correct parameter names are matched the corresponding value
        j = 0
        for param_name, _ in domain.items():
            if self.parameters["material_parameters"].get(param_name):
                domain[param_name] = sampled_parameters[:,j].astype(self.parameters["material_parameters"][param_name].get("type"))
            else:
                domain[param_name] = sampled_parameters[:,j].astype(self.parameters["geometrical_parameters"][param_name].get("type"))
            j += 1

        # initialize the targets and the statistical parameters 
        targets = collections.defaultdict(list)
        features = collections.defaultdict(list)

        # get the executable
        current_driver = self.driver
        driver_name = current_driver["driver_params"].get("executable_name")

        try:
            current_exec = self.global_settings["exe_paths"][driver_name]
        except:
            raise FileNotFoundError(f"Executable {driver_name} does not exist!")

        # iterate the simulations
        for i in range(self.num_simulations):
            # create the instance for the rough surface
            rough_surf = RoughSurface(self.global_settings["output_dir"], domain["n"][i], domain["H"][i], domain["g0"][i], i, domain["lato"][i])
            # generate the rough surface  
            surface_path = rough_surf.generate_surface_RMD()

            # generate the input file for the BEM executable
            bem_inp_file = self.generate_json(surface_path, i, domain)
            # run the BEM executable
            self.call_executable(bem_inp_file, current_exec, i)

            if not os.path.exists((self.global_settings["output_dir"] + '/result_force_surface_' + str(i) + '.dat')):
                sim_fail = f'''
***********************************************************************
-------- Simulation {i} failed, jumping to the next simulation --------
***********************************************************************
                '''
                print(sim_fail)
                continue

            if(self.result_description.get("write_results")):
                # calculate the effective contact area and traction after BEM simulation is run
                targets = self.post_process_bem(i, domain["n"][i], targets)
                # calculate the Statistical properties
                features = rough_surf.random_postprocess(features)

                for key, _ in domain.items():
                    features[key].append(domain[key][i])
                # features["H"].append(domain["H"][i])

        if(self.result_description.get("write_results")):
            # output the final combined results into a file
            final_results = self.write_final_results(targets, features)
            # plot fancy results
            self.save_plot(final_results) 

    def get_parameters(self):
        '''
        Obtains the parameters from the input file and generates the range for each parameter
        '''
        
        mat_params = self.parameters["material_parameters"]
        geo_params = self.parameters["geometrical_parameters"]

        domain = collections.OrderedDict()

        for param, value in mat_params.items():
            if value["size"] != 1:
                domain[param] = np.linspace(value["distribution_parameter"][0],value["distribution_parameter"][1],value["size"]).astype(value["type"])
            else:
                domain[param] = [value["distribution_parameter"]]
        
        for param, value in geo_params.items():
            if value["size"] != 1:
                domain[param] = np.linspace(value["distribution_parameter"][0],value["distribution_parameter"][1],value["size"]).astype(value["type"])
            else:
                domain[param] = [value["distribution_parameter"]]
        
        return domain



    def generate_json(self, surface_path, n_iter, domain):
        '''
        Generates the json input file for the BEM executable

        Args
        ---
        surface_path : str
            shows BEM executable where to find the rough surface z file
        n_iter : int
            the corresponding simulation number to distiguish the BEM output files 
        '''

        data = {
                    "z_file_path" : surface_path,
                    "parameters" : 
                    {
                        "material_parameters" : 
                        {
                            "E1"  : float(domain["E1"][n_iter]),
                            "nu1" : float(domain["nu1"][n_iter]),
                            "E2"  : float(domain["E2"][n_iter]),
                            "nu2" : float(domain["nu2"][n_iter])
                        },
                        "geometrical_parameters" : 
                        {
                            "lato" :  float(domain["lato"][n_iter]),
                            "n" :     int(domain["n"][n_iter]),
                            "Delta" : float(domain["Delta"][n_iter]), 
                            "errf" :  float(domain["errf"][n_iter]),
                            "tol" :   float(domain["tol"][n_iter])
                        }
                    }
                }

        path_out = self.global_settings["output_dir"] 
        bem_inp_file = path_out + '/input_bem_' + str(n_iter) + ".json"
    
        with open(bem_inp_file, 'w') as out_file:
            json.dump(data, out_file, indent=4)
        
        return bem_inp_file

    def call_executable(self, bem_inp_file, current_exec, i):
        '''
        Calls the BEM executable

        Args
        ---
        bem_inp_file : str
            the BEM input file in json format 
        '''   

        args = [current_exec,bem_inp_file]

        simulation_start = f'''
-----------------------------------------------------------------------------------------
**************************** BEM Simulation started *************************************
-----------------------------------------------------------------------------------------
Simulation number: -{i}-
-----------------------------------------------------------------------------------------
        '''
        print(simulation_start)

        subprocess.call(args)

    def post_process_bem(self, n_iter, n, targets):
        '''
        Calculates the effective contact area and corresponding traction force (targets) after the BEM simulation is run

        Args
        ---
        n_iter: int
            the corresponding simulation number to distiguish the BEM output files
        n: int
            the parameter dimension of the problem (comes from json input file)
        targets: dict
            the value-key pairs containing the total effective contact area and corresponding traction force 
        '''

        file_name = self.global_settings["output_dir"] + '/result_force_surface_' + str(n_iter) + '.dat'

        # the number of points in contact
        n_contact = np.genfromtxt(file_name,delimiter=";")[:-1].size
        # total contact force
        total_force = np.genfromtxt(file_name,delimiter=";")[:-1].sum()

        # effective contact area in percent %
        eff_area = n_contact * (1/(2**n + 1))**2 * 100

        targets["total_force"].append(total_force)
        targets["total_cont_area"].append(eff_area)
        
        return targets

    def write_final_results(self, targets , features):
        '''
        Writes the final results as a DataFrame (targets and statistical features) into a file called simulation_output.dat

        Args
        ---
        targets: dict
            the key-value pair containing the total effective contact area and corresponding traction force
        features: dict
            the key-value pair containing the statistical properties of the rough surface  
        '''
        simulation_file_name = "_".join(self.global_settings["experiment_name"].split())
        simulation_file = self.global_settings["output_dir"] + '/simulation_output_' + simulation_file_name + '.dat'

        print(f'''
-------------------------------------------------------------------------
Simulation inputs/ouputs are stored in {simulation_file}
-------------------------------------------------------------------------
        ''')
        
        # copy features into targets (merge two dicts)
        targets.update(features)

        # transform targets dict into a dataframe
        df = pd.DataFrame.from_dict(targets)

        # store the results
        df.to_csv(simulation_file, index=False, sep="\t", float_format='%.5f')

        return simulation_file
    
    def save_plot(self, final_results):
        '''
        Plots of the final results and stores them

        Args
        ---
        final_results: str
            the file contains the final results
        '''

        df = pd.read_csv(final_results, sep="\t")

        fig, axes = plt.subplots(4,16, figsize=(45,10))

        for i in range(4):
            for j in range(16):
                ax_obj = sns.scatterplot(ax=axes[i,j], y=df.iloc[:,i//2] ,x=df.iloc[:,(i % 2)*16+j+2], data=df)
                ax_obj.set(ylabel=df.iloc[:,i//2].name, xlabel=df.iloc[:,(i % 2)*16+j+2].name)

        plt.tight_layout()
        figure_name = self.global_settings["output_dir"] + '/simulation_results.png'
        fig.savefig(figure_name)