import numpy as np
from numpy.random import seed
from numpy import random as rnd
import os
import subprocess

from .iterator import Iterator

class BaciFemIterator(Iterator): 
    
    def __init__(self, input_dir, result_description, driver, global_settings):
        super(BaciFemIterator, self).__init__(None, global_settings)
        self.result_description = result_description
        self.driver = driver
        self.input_dir = input_dir
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("experiment_name"))

        method_options = config["method"]["method_options"]
        result_description = method_options.get("result_description", None)
        input_dir = method_options.get("input_dir", None)

        driver = config.get("driver", None)
        global_settings = config.get("global_settings", None)


        return cls(input_dir, result_description, driver, global_settings)

    def run_simulation(self):

        my_driver = self.driver       
        solver_name = my_driver["driver_params"]["executable_type"].get("solver", None)
        post_processing_name = my_driver["driver_params"]["executable_type"].get("post_processing", None)
        output_prefix = my_driver["driver_params"]["output_prefix"]

        if solver_name == None:
            raise Exception("solver is not defined")
#------------------------------------------------
        current_driver = self.driver
        solver_name = current_driver["driver_params"]["executable_type"].get("solver", None)
        post_processing_name = current_driver["driver_params"]["executable_type"].get("post_processing", None)

        try:
            current_solver = self.global_settings["exe_paths"].get(solver_name)
        except:
            raise FileNotFoundError(f"Executable {solver_name} does not exist!") 
        
        try:
            current_post_proces = self.global_settings["exe_paths"].get(post_processing_name)
        except:
            raise FileNotFoundError(f"Executable {post_processing_name} does not exist!") 
        

        n_iter = 0
        for inp_file in os.listdir(self.input_dir):

            output_prefix = output_prefix + "_" + str(n_iter)

            # call solver      
            exec_options = self.get_solv_options(inp_file, output_prefix, n_iter)
            self.call_executable(current_solver, exec_options)

            # call post_processing
            exec_options = self.get_post_options(output_prefix)
            self.call_executable(current_post_proces, exec_options)
 
            n_iter += 1
            

    def get_solv_options(self, inp_file, output_prefix, n_iter):
        '''
        args for baci-release -> input_dir/datfile ouput_dir/output_prefix
        '''
        dat_file = os.path.join(self.input_dir, inp_file)
        output_file = os.path.join(self.global_settings["output_dir"], output_prefix)
        args = [dat_file, output_file]
        return args

    def get_post_options(self, output_prefix):
        '''
        args for post_drt_ensight
        '''
        output_file = os.path.join(self.global_settings["output_dir"], output_prefix)
        prefix = '--file='
        args = [prefix + output_file]
        return args


    def call_executable(self, exec_type, args):
        

        args.insert(0, exec_type)

        # subprocess.call(args)
        process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE)
        # rc = process.poll()
        stdout, stderr = process.communicate()
        #print(stdout)
        # out = subprocess.check_output(args)

        #while True:
        #    line = process.stdout.readline().rstrip()
        #    if not line:
        #        break
        #    yield line