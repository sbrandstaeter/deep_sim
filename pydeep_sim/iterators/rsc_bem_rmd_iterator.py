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

class RoughSurfaceBemRMDIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, num_simulations, result_description, driver, parameters, global_settings):
        super(RoughSurfaceBemRMDIterator, self).__init__(None, global_settings)
        self.num_simulations = num_simulations
        self.result_description = result_description
        self.driver = driver
        self.parameters = parameters
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("global_settings", None)["experiment_name"])

        method_options = config["method"]["method_options"]
        num_simulations = method_options.get("num_simulations", None)
        result_description = method_options.get("result_description", None)
        
        driver = config.get("driver", None)
        parameters = config.get("parameters", None)
        global_settings = config.get("global_settings", None)

        return cls(num_simulations, result_description, driver, parameters, global_settings)

    def run_simulation(self):
        '''
        Run the BEM simulation 
        '''       

        # obtain the parameters to run the simulation
        n_global = self.parameters["geometrical_parameters"]["n"].get("distribution_parameter")
        H_global = self.parameters["geometrical_parameters"]["H"].get("distribution_parameter")
        g0_global = self.parameters["geometrical_parameters"]["g0"].get("distribution_parameter")
        H_range = np.linspace(H_global[0],H_global[1],self.num_simulations)
        lato = self.parameters["geometrical_parameters"]["lato"]["distribution_parameter"]

        # initialize the targets and the statistical parameters 
        targets = collections.defaultdict(list)
        statistical_properties = collections.defaultdict(list)

        #
        current_driver = self.driver
        driver_name = current_driver["driver_params"].get("executable_name")

        try:
            current_exec = self.global_settings["exe_paths"]["driver_name"]
        except:
            raise FileNotFoundError(f"Executable {driver_name} does not exist!")

        # repeat the simulations
        for i in range(self.num_simulations):
            # create the instance for the rough surface
            rough_surf = RoughSurface(self.global_settings["output_dir"], n_global, H_range[i], g0_global, i, lato)
            # generate the rough surface  
            surface_path = rough_surf.generate_surface_RMD()

            # generate the input file for the BEM executable
            bem_inp_file = self.generate_json(surface_path, i)
            # run the BEM executable
            self.call_executable(bem_inp_file, current_exec, i)

            if not os.path.exists((self.global_settings["output_dir"] + '/result_force_surface_' + str(i) + '.dat')):
                sim_fail = f'''
**************************************************************
---- Simulation {i} failed, jump to the next simulation ------
**************************************************************
                '''
                print(sim_fail)
                continue

            if(self.result_description.get("write_results")):
                # calculate the effective contact area and traction after BEM simulation is run
                targets = self.post_process_bem(i, n_global, targets)
                # calculate the Statistical properties
                statistical_properties = rough_surf.random_postprocess(statistical_properties)
                statistical_properties["H"].append(H_range[i])

        if(self.result_description.get("write_results")):
            # output the final combined results into a file
            final_results = self.write_final_results(targets, statistical_properties)
            # plot fancy results
            self.save_plot(final_results) 

    def generate_json(self,surface_path,n_iter):
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
                            "E1"  : self.parameters["material_parameters"]["E1"]["distribution_parameter"],
                            "nu1" : self.parameters["material_parameters"]["nu1"]["distribution_parameter"],
                            "E2"  : self.parameters["material_parameters"]["E2"]["distribution_parameter"],
                            "nu2" : self.parameters["material_parameters"]["nu2"]["distribution_parameter"]
                        },
                        "geometrical_parameters" : 
                        {
                            "lato" :  self.parameters["geometrical_parameters"]["lato"]["distribution_parameter"],
                            "n" :     self.parameters["geometrical_parameters"]["n"]["distribution_parameter"],
                            "Delta" : self.parameters["geometrical_parameters"]["Delta"]["distribution_parameter"], 
                            "errf" :  self.parameters["geometrical_parameters"]["errf"]["distribution_parameter"],
                            "tol" :   self.parameters["geometrical_parameters"]["tol"]["distribution_parameter"]
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

    def write_final_results(self, targets , statistical_properties):
        '''
        Writes the final results as a DataFrame (targets and statistical features) into a file called simulation_output.dat

        Args
        ---
        targets: dict
            the key-value pair containing the total effective contact area and corresponding traction force
        statistical_properties: dict
            the key-value pair containing the statistical properties of the rough surface  
        '''

        simulation_file = self.global_settings["output_dir"] + '/simulation_output' + '.dat'

        print(f'''
-------------------------------------------------------------------------
Simulation inputs/ouputs are stored in {simulation_file}
-------------------------------------------------------------------------
        ''')
        
        # copy statistical_properties into targets (merge two dicts)
        targets.update(statistical_properties)

        # transform targets dict into a dataframe
        df = pd.DataFrame.from_dict(targets)

        # store the results
        df.to_csv(simulation_file, index=False, sep="\t", float_format='%.3f')

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

        fig, axes = plt.subplots(4,11, figsize=(35,10))

        for i in range(4):
            for j in range(11):
                ax_obj = sns.scatterplot(ax=axes[i,j], y=df.iloc[:,i//2] ,x=df.iloc[:,(i % 2)*11+j+2], data=df)
                ax_obj.set(ylabel=df.iloc[:,i//2].name, xlabel=df.iloc[:,(i % 2)*11+j+2].name)

        plt.tight_layout()
        figure_name = self.global_settings["output_dir"] + '/simulation_results.png'
        fig.savefig(figure_name)