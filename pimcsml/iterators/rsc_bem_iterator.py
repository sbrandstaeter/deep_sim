import numpy as np
from numpy.random import seed
from numpy import random as rnd
import subprocess

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
        
        n_global = self.parameters["geometrical_parameters"]["n"]
        H_global = self.parameters["geometrical_parameters"]["H"]

        n = n_global["distribution_parameter"]

        start = H_global["distribution_parameter"][0]
        end = H_global["distribution_parameter"][1]
        seeds = H_global["size"]

        H_iterator = np.linspace(start, end, seeds)

        for i in range(self.num_simulations):
            
            H = H_iterator[i]
#            breakpoint()
            surface_path = self.generate_2D_surface(n, H, i)
            self.call_executable(surface_path)
            print(surface_path)

    def generate_2D_surface(self,n,H,iter):
        '''
        creates the 2D surfaces
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

        zref = self.parameters["geometrical_parameters"]["zref"].get("distribution_parameter")
        scalefactor = zref/(np.max(z)-np.mean(z))
        z = z*scalefactor
        z = z-(np.min(z))

        path_out = self.global_settings["output_dir"]
        full_path = path_out + "/surface_" + str(iter) + ".dat"
        np.savetxt(full_path, z, delimiter=';')
        
        return full_path

    def call_executable(self,surface_path):
        
        my_driver = self.driver
        my_exec = self.global_settings["executable_path"] + "/"  + my_driver["driver_params"].get("executable_name")
        args = [my_exec,surface_path]

        subprocess.call(args)