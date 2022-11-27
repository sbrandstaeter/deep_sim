import numpy as np
from numpy.random import seed
from numpy import dtype, random as rnd
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from .iterator import Iterator
from ..pinns.pinn_model import PinnModel
from ..sampling.sampling import Sampling

class PinnIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, num_simulations, result_description, driver, parameters, sampling, global_settings):
        super(PinnIterator, self).__init__(None, global_settings)
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
            search_flag = True
            for param_type, _ in self.parameters.items():
                if self.parameters[param_type].get(param_name) and search_flag:
                    domain[param_name] = sampled_parameters[:,j].astype(self.parameters[param_type][param_name].get("type"))
                    search_flag = False
                    j += 1

        # get the pinn model
        current_driver = self.driver

        # iterate the simulations
        for simulation_number in range(self.num_simulations):
            
            # create the instance for the pinn
            pinn_object = PinnModel(self.global_settings["output_dir"], current_driver, domain, simulation_number)
            # generate the rough surface  
            model = pinn_object.build_model()
            # train the model
            losshistory, train_state, model = pinn_object.run(model)

            if(self.result_description.get("write_results")):
                if self.driver["driver_options"].get("model_output"):
                    pinn_object.model_output(losshistory, train_state, model, self.result_description, simulation_number)
                # # calculate the effective contact area and traction after BEM simulation is run
                # targets = self.post_process_bem(i, domain["n"][i], targets)
                # # calculate the Statistical properties
                # features = rough_surf.random_postprocess(features)

                # for key, _ in domain.items():
                #     features[key].append(domain[key][i])
        
        # if(self.result_description.get("write_results")):
        #     # output the final combined results into a file
        #     final_results = self.write_final_results(targets, features)
        #     # plot fancy results
        #     self.save_plot(final_results) 

    def get_parameters(self):
        '''
        Obtain the parameters from the input file and generates the range for each parameter
        '''

        domain = collections.OrderedDict()
        
        
        def get_domain(param, param_options, domain):
            if param_options["type"] == "str":
                if isinstance(param_options["distribution_parameter"], list):
                    domain[param] = param_options["distribution_parameter"]
                else:
                    domain[param] = [param_options["distribution_parameter"]]
            else:
                if param_options["size"] != 1:
                    domain[param] = np.linspace(param_options["distribution_parameter"][0],param_options["distribution_parameter"][1],param_options["size"]).astype(param_options["type"])
                else:
                    domain[param] = [param_options["distribution_parameter"]]
            return domain
        
        for param_type, params in self.parameters.items():
            for param, param_options in params.items():
                if not param in domain.keys():
                    domain = get_domain(param, param_options, domain)
                else:
                    raise NameError(f"Parameter {param} in {param_type} is already in previous parameters. Change the name.")
        
        return domain

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