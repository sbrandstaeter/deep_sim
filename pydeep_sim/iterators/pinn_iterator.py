import numpy as np
import collections
import warnings
import subprocess
import sys

from .iterator import Iterator
from ..pinns.pinns import Pinns
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
        Run the Physics-Informed Neural Network
        '''       
        # sampling methods require the ranges of the parameters
        # first get the parameters from the input file
        domain = self.get_parameters()

        # create sampling instance
        sample_model = Sampling(domain=list(domain.values()), 
                        n_samples=self.num_simulations, 
                        sampling_name=self.sampling["sampling_name"], 
                        sampling_options=self.sampling.get("sampling_options"))

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
        
        # input file name
        pinn_model_file_name = self.result_description.get("pinn_model_file_name","default_pinn_model")

        # iterate the simulations
        for simulation_number in range(self.num_simulations):
            
            pinn_model_file_name_ = pinn_model_file_name + f"_{simulation_number}"
            # generate the model class 
            pinn_model_class = Pinns.from_driver_build_model(current_driver)
            # create the instance for the pinn
            pinn_object = pinn_model_class(self.global_settings["output_dir"], self.result_description, current_driver, domain, simulation_number, pinn_model_file_name_)
            # generate the input file
            pinn_model_input_file = pinn_object.generate_input_file()
            # run the input file
            output_file = self.run_input_file(pinn_model_input_file, simulation_number)
            # write results
            if(self.result_description.get("write_results")):
                if self.result_description.get("output_options"):
                    pinn_object.model_output(output_file)

    def get_parameters(self):
        '''
        Obtain the parameters from the input file and generates the range for each parameter.
        
        Returns
        -------
        domain : dict
            generated domain from input file
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
                    if isinstance(param_options["distribution_parameter"], list):
                        domain[param] = param_options["distribution_parameter"]
                    else:
                        domain[param] = [param_options["distribution_parameter"]]
                    if (param_options["type"] == "int") and (not isinstance(param_options["distribution_parameter"], int)):
                        warnings.warn(f"Parameter value for {domain[param]} is not int! So it will be set as integer.")
                        domain[param] = int(domain[param])
            return domain
        
        for param_type, params in self.parameters.items():
            for param, param_options in params.items():
                if not param in domain.keys():
                    domain = get_domain(param, param_options, domain)
                else:
                    raise NameError(f"Parameter {param} in {param_type} is already in previous parameters. Change the name.")
        
        return domain
    
    def run_input_file(self, input_file, simulation_number):
        
        output_file = self.global_settings["output_dir"] + "/" + self.result_description["pinn_model_file_name"] + "_" + str(simulation_number) + "_results"
        
        f = open(output_file, "w")
        subprocess.run([sys.executable,input_file], stdout=f)
        f.close()
        
        return output_file