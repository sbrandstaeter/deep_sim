import numpy as np
from numpy.random import seed
from numpy import dtype, random as rnd
import subprocess
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from .iterator import Iterator
from ..rough_surface.rough_surface import RoughSurface

class RoughSurfaceBemRMDIterator(Iterator): 
    
    def __init__(self, num_simulations, result_description, driver, parameters, global_settings):
        super(RoughSurfaceBemRMDIterator, self).__init__(None, global_settings)
        self.num_simulations = num_simulations
        self.result_description = result_description
        self.driver = driver
        self.parameters = parameters
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("experiment_name"))

        method_options = config["method"]["method_options"]
        num_simulations = method_options.get("num_simulations", None)
        result_description = method_options.get("result_description", None)
        
        driver = config.get("driver", None)
        parameters = config.get("parameters", None)
        global_settings = config.get("global_settings", None)

        return cls(num_simulations, result_description, driver, parameters, global_settings)

    def run_simulation(self):       

        if(self.result_description.get("write_results")):
            simulation_file = self.global_settings["output_dir"] + '/simulation_output' + '.dat'
            print("Simulation inputs/ouputs are stored in simulation_output.dat folder")
            
            with open(simulation_file, "a") as myfile:
                myfile.write("H" + "\t" + 
                             "mean" + "\t" +
                            "std" + "\t" +
                            "total_force" + "\t" +
                            "total_area"  + "\n")
            myfile.close()

        n_global = self.parameters["geometrical_parameters"]["n"].get("distribution_parameter")
        H_global = self.parameters["geometrical_parameters"]["H"].get("distribution_parameter")
        g0_global = self.parameters["geometrical_parameters"]["g0"].get("distribution_parameter")
        H_range = np.linspace(H_global[0],H_global[1],self.num_simulations)
        lato = self.parameters["geometrical_parameters"]["lato"]["distribution_parameter"]


        for i in range(self.num_simulations):

            rough_surf = RoughSurface(self.global_settings["output_dir"], n_global, H_range[i], g0_global, i, lato) # create the instance for the rough surface
            surface_path = rough_surf.generate_surface_RMD()

            bem_inp_file = self.generate_json(surface_path, i)
            self.call_executable(bem_inp_file)
            targets = self.calculate_area(i, n_global)

            if(self.result_description.get("write_results")):
                statistical_properties = rough_surf.random_postprocess()
                final_results = self.write_final_result(H_range[i],targets, simulation_file, statistical_properties)

        self.save_plot(final_results) 

    def generate_json(self,surface_path,n_iter):

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

    def call_executable(self,bem_inp_file):
        
        my_driver = self.driver
        my_exec = self.global_settings["executable_path"] + "/"  + my_driver["driver_params"].get("executable_name")
        args = [my_exec,bem_inp_file]

        subprocess.call(args)

    def calculate_area(self, n_iter, n):

        # fomula -> area = nf * delta**2 / lato**2 *100
        file_name = self.global_settings["output_dir"] + '/result_force_surface_' + str(n_iter) + '.dat'
        
        file1 = open(file_name, 'r')
        Lines = file1.readlines()
        
        n_contact = 0 # the number of uncontacted points
        total_force = 0 
        i = 0
        while not Lines[0].split(';')[i] == "":
            n_contact += 1
            total_force += float(Lines[0].split(';')[i])
            i += 1 

        file1.close()

        eff_area = n_contact * (1/(2**n + 1))**2 * 100

        targets = {"total_force":total_force,
                   "total_cont_area":eff_area}
        
        return targets

    def write_final_result(self, H, targets, simulation_file, statistical_properties):
        
        def format(value):
            return "%.3f" % value

        with open(simulation_file, "a") as myfile:
            myfile.write(str(format(H))+ "\t" + 
                         str(format(statistical_properties["z_mean"])) + "\t" +
                         str(format(statistical_properties["z_rms"])) + "\t" +
                         str(format(total_force)) + "\t" +
                         str(format(area))  + "\n")
        
        myfile.close()
        
        return simulation_file
    
    def save_plot(self, final_results):

        df = pd.read_csv(final_results, sep="\t")

        plt.clf()
        plt.scatter(df["H"],df["total_area"])
        plt.xlabel("Hurst exponent")
        plt.ylabel("Totol effective contact area in %")
        figure_name = self.global_settings["output_dir"] + '/area_vs_hurst.png'
        plt.savefig(figure_name)

        plt.clf()
        plt.scatter(df["mean"],df["total_area"])
        plt.xlabel("the mean value of z")
        plt.ylabel("Total effective contact area in %")
        figure_name = self.global_settings["output_dir"] + '/area_vs_mean.png'
        plt.savefig(figure_name)

        plt.clf()
        plt.scatter(df["std"],df["total_area"])
        plt.xlabel("the std of z")
        plt.ylabel("Total effective contact area in %")
        figure_name = self.global_settings["output_dir"] + '/area_vs_std.png'
        plt.savefig(figure_name)