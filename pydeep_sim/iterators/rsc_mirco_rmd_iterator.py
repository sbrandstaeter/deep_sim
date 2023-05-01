import numpy as np
import subprocess
import os
import math
import pandas as pd
import collections

from .iterator import Iterator
from ..rough_surface.rough_surface import RoughSurface
from ..sampling.sampling import Sampling

class RoughSurfaceMIRCORMDIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, num_simulations, result_description, driver, parameters, sampling, global_settings):
        super(RoughSurfaceMIRCORMDIterator, self).__init__(None, global_settings)
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
                        sampling_options=self.sampling.get("sampling_options"))

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
        # targets = collections.defaultdict(list)
        # features = collections.defaultdict(list)

        # get the executable
        current_driver = self.driver
        driver_name = current_driver["driver_params"].get("executable_name")
        driver_options = current_driver["driver_params"].get("driver_options")

        try:
            current_exec = self.global_settings["exe_paths"][driver_name]
        except:
            raise FileNotFoundError(f"Executable {driver_name} does not exist!")

        # iterate the simulations
        for i in range(self.num_simulations):
            targets = collections.defaultdict(list)
            features = collections.defaultdict(list)
            
            sim_number_divider = int(math.log10(self.num_simulations))+1
            file_tail = f"{i:0{sim_number_divider}}"
            
            # create the instance for the rough surface
            rough_surf = RoughSurface(self.global_settings["output_dir"],
                                      file_tail, 
                                      domain["Resolution"][i], 
                                      domain["HurstExponent"][i], 
                                      domain["InitialTopologyStdDeviation"][i], 
                                      i, 
                                      domain["LateralLength"][i])

            
            # generate the rough surface  
            surface_path = rough_surf.generate_surface_RMD()
            # generate the input file for MIRCO and run it
            path_to_xml_input = self.generate_xml_input(surface_path, i, file_tail, domain)
            time_out = self.call_executable(path_to_xml_input, driver_options, current_exec, file_tail)
            
            # check if simulation failed due to maximum waiting time
            if time_out:
                self.warning_fail(file_tail, driver_options["output_file_prefix"], fail=False)
                continue
            
            # check if simulation is failed due to no convergence
            not_fail_flag = os.path.exists((self.global_settings["output_dir"] + '/' + driver_options["output_file_prefix"] + "_" + file_tail + "_info.csv"))
            if not not_fail_flag:
                self.warning_fail(file_tail, driver_options["output_file_prefix"])
                continue

            if(self.result_description.get("write_results")):
                info_output_file = self.global_settings["output_dir"] + '/' + driver_options["output_file_prefix"] + "_" + file_tail + "_info.csv"
                # obtain simulation results after MIRCO simulation is finished
                targets = self.get_results(info_output_file, targets)
                # calculate the Statistical properties
                features = rough_surf.random_postprocess(surface_path, features)
                features["topology_file"] = os.path.splitext(os.path.basename(surface_path))[0]

                for key, _ in domain.items():
                    features[key].append(domain[key][i])
                
                final_results = self.write_final_results(targets, features, i)
                # features["H"].append(domain["H"][i])
        
        print(f'''
-------------------------------------------------------------------------
Simulation inputs/ouputs are stored in {final_results}
-------------------------------------------------------------------------
        ''')
                 
    def warning_fail(self, file_tail, output_file_prefix, fail=True):
        flag_print = "failed"  if fail else "terminated due to run time"
        sim_fail = f'''
*************************************************************************************
- Simulation {file_tail} {flag_print}, jumping to the next simulation -
*************************************************************************************
                '''
        print(sim_fail)
        # generate log file
        log_file = self.global_settings["output_dir"] + "/" +output_file_prefix + ".log"
        mode = 'a' if os.path.exists(log_file) else 'w'
        with open(log_file , mode) as f :
            if fail:
                f.write(f"Simulation {file_tail} failed.\n")
            else:
                f.write(f"Simulation {file_tail} interrupted.\n")
    
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

    def generate_xml_input(self, surface_path, n_iter, file_tail, domain):
        '''
        Generates the XML input file for the MIRCO executable

        Args
        ---
        surface_path : str
            path to rough surface mesh
        n_iter : int
            the corresponding simulation number to distiguish the BEM output files 
        '''
        path_out = self.global_settings["output_dir"]
        
        path_to_xml_input = path_out + '/input_' + file_tail + ".xml"
        
        # Convert True to true or False to false
        converter_bool = lambda x: str(x).lower() if isinstance(x, bool) else  x
        
        # Convert float to double
        converter_f_d = lambda x: "double" if x=="float" else x
        
        xml_input_file = open(path_to_xml_input, "w")
        
        # add start header
        xml_input_file.write("<ParameterList>\n")
        
        # add simulation parameters from input
        for param_name, param_options in self.parameters["simulation_parameters"].items():
            data_type = param_options["type"]
            data_value = converter_bool(param_options["distribution_parameter"])
            line_ = f'  <Parameter name="{param_name}" type="{data_type}" value="{data_value}"/>\n'
            xml_input_file.write(line_)
        
        # add topology file path
        xml_input_file.write(f'  <Parameter name="TopologyFilePath" type="string" value="{surface_path}"/>\n')
        xml_input_file.write("\n")
        
        # add material parameters
        xml_input_file.write('  <ParameterList name="parameters">\n')
        xml_input_file.write('    <ParameterList name="material_parameters">\n')
        for param_name, param_options in self.parameters["material_parameters"].items():
            data_type = converter_f_d(param_options["type"])
            data_value = domain[param_name][n_iter]
            line_ = f'      <Parameter name="{param_name}" type="{data_type}" value="{data_value}"/>\n'
            xml_input_file.write(line_)
        # add closing
        xml_input_file.write("    </ParameterList>\n")
        xml_input_file.write("\n")
        
        # add geometrical parameters
        xml_input_file.write('    <ParameterList name="geometrical_parameters">\n')
        for param_name, param_options in self.parameters["geometrical_parameters"].items():
            data_type = converter_f_d(param_options["type"])
            data_value = domain[param_name][n_iter]
            line_ = f'      <Parameter name="{param_name}" type="{data_type}" value="{data_value}"/>\n'
            xml_input_file.write(line_)
        # add closing
        xml_input_file.write("    </ParameterList>\n")
        xml_input_file.write("\n")
        
        # add closing
        xml_input_file.write("  </ParameterList>\n")
        xml_input_file.write("</ParameterList>\n")
        xml_input_file.close()
        
        
        return path_to_xml_input

    def call_executable(self, path_to_xml_input, driver_options, current_exec, file_tail):
        '''
        Calls the MIRCO executable

        Args
        ---
        path_to_xml_input : str
            the MIRCO input file in XML format 
        output_file_prefix : str
            output prefix
        current_exec : EXE
            current executable  
        '''   
        output_file_prefix = driver_options["output_file_prefix"] + "_" + file_tail
        args = [current_exec, path_to_xml_input, output_file_prefix]

        simulation_start = f'''
-----------------------------------------------------------------------------------------
**************************** MIRCO simulation started *************************************
-----------------------------------------------------------------------------------------
Simulation number: -{file_tail}-
-----------------------------------------------------------------------------------------
        '''
        print(simulation_start)
        try:
            subprocess.call(args, timeout=driver_options.get("max_run_time"))
            time_out = False
        except:
            time_out = True
        
        return time_out
    
    def get_results(self, mirco_output_file, targets):
        '''
        Obtain simulation results after MIRCO simulation is finished.

        Args
        ---
        mirco_output_file: file
            the corresponding simulation number to distiguish the BEM output files
        targets: dict
            the value-key pairs containing the total effective contact area and corresponding traction force 
        '''
        df = pd.read_csv(mirco_output_file, delimiter="\t")
        # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        for col in df.columns:
            targets[col].append(df[col][0]) 
        
        return targets
    
    def write_final_results(self, targets , features, n_iter):
        '''
        Writes the final results as a DataFrame (targets and statistical features) into a file called simulation_output.dat

        Args
        ---
        targets: dict
            the key-value pair containing the total effective contact area and corresponding traction force
        features: dict
            the key-value pair containing the statistical properties of the rough surface  
        '''
        final_results = "_".join(self.global_settings["experiment_name"].split())
        final_results = self.global_settings["output_dir"] + '/simulation_output_' + final_results + '.csv'
        
        # copy features into targets (merge two dicts)
        targets.update(features)

        # transform targets dict into a dataframe
        df = pd.DataFrame.from_dict(targets)
        
        if n_iter == 0:
            df.to_csv(final_results, index=False, sep="\t", float_format='%.5f')
        else:
            df.to_csv(final_results, mode="a", index=False, header=False, sep="\t", float_format='%.5f')
            
        return final_results
            