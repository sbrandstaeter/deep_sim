import numpy as np
from numpy.random import seed
from numpy import random as rnd
import subprocess
import os
import json

from .iterator import Iterator

class RoughSurfaceBemIterator(Iterator): 
    
    def __init__(self, num_simulations, result_description, driver, parameters, global_settings):
        super(RoughSurfaceBemIterator, self).__init__(None, global_settings)
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
        
        n_global = self.parameters["geometrical_parameters"]["n"].get("distribution_parameter")
        H_global = self.parameters["geometrical_parameters"]["H"].get("distribution_parameter")
        zref_global = self.parameters["geometrical_parameters"]["zref"].get("distribution_parameter")

        for i in range(self.num_simulations):
            
            surface_path = self.generate_2D_surface(n_global, H_global, zref_global, i)
            bem_inp_file = self.generate_json(surface_path, i)
            self.call_executable(bem_inp_file)


    def generate_2D_surface(self, n, H, zref, iter):
        '''
        creates the 2D surfaces using RMD (Random Midpoint Distribution)
        '''
        N = 2**n     
        z = np.zeros([N+1,N+1])

        alpha = 1 / np.sqrt(0.09)

        D = N
        d = N//2

        for _ in range(n):
            alpha=alpha/np.sqrt(2)**H
            
            for j in range(d,N-d+1,D):
                for k in range(d,N-d+1,D):
                    z[j,k] =  (z[j+d,k+d]+z[j+d,k-d]+z[j-d,k+d]+z[j-d,k-d])/4+alpha*rnd.randn()
            
            alpha=alpha/np.sqrt(2)**H
            
            for j in range(d,N-d+1,D):
                z[j,0]=(z[j+d,0]+z[j-d,0]+z[j,d])/3+alpha*rnd.randn()
                z[j,N]=(z[j+d,N]+z[j-d,N]+z[j,N-d])/3+alpha*rnd.randn()
                z[0,j]=(z[0,j+d]+z[0,j-d]+z[d,j])/3+alpha*rnd.randn()
                z[N,j]=(z[N,j+d]+z[N,j-d]+z[N-d,j])/3+alpha*rnd.randn()
    
            for j in range(d,N-d+1,D):
                for k in range(D,N-d+1,D):
                    z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()
            
            for j in range(D,N-d+1,D):
                for k in range(d,N-d+1,D):
                    z[j,k]=(z[j,k+d]+z[j,k-d]+z[j+d,k]+z[j-d,k])/4+alpha*rnd.randn()


            D = D//2
            d = d//2 

        scalefactor = zref/(np.max(z)-np.mean(z))
        z = z*scalefactor
        z = z-(np.min(z))

        path_out = self.global_settings["output_dir"]
        full_path = path_out + "/surface_" + str(iter) + ".dat"
        np.savetxt(full_path, z, delimiter=';', fmt="%15.5e")
        
        return full_path

    def generate_json(self,surface_path,iter):

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
        bem_inp_file = path_out + '/input_bem_' + str(iter) + ".json"
    
        with open(bem_inp_file, 'w') as out_file:
            json.dump(data, out_file, indent=4)
        
        return bem_inp_file

    def call_executable(self,bem_inp_file):
        
        my_driver = self.driver
        my_exec = self.global_settings["executable_path"] + "/"  + my_driver["driver_params"].get("executable_name")
        args = [my_exec,bem_inp_file]

        subprocess.call(args)