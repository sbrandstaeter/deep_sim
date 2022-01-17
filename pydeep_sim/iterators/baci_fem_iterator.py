from typing import OrderedDict
import numpy as np
from numpy.random import seed
from numpy import random as rnd
import os
import subprocess
import pandas as pd
from paraview import simple
# from vtk.util import numpy_support as VN

from paraview.simple import *

from .iterator import Iterator

class BaciFemIterator(Iterator):
    '''
    A class to run FEM simulations using BACI

    Attributes
    result_description : dict
        properties of the result file
    driver : dict
        options for baci-release and post-drt-ensight 
    global_settings:  dict
        contains output directory and experiment name independent of the given problem
    ''' 
    
    def __init__(self, input_dir, result_description, driver, global_settings):
        super(BaciFemIterator, self).__init__(None, global_settings)
        self.result_description = result_description
        self.driver = driver
        self.input_dir = input_dir
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):
        '''
        Creates the iterator from problem description

        Parameters
        ----------
        config : dict
            contains the input parameters
        iterator_name : str , optional
            iterator to be run

        Returns
        -------
        cls(num_simulations, result_description, driver, global_settings)
            a class object of the chosen iterator
        '''

        print(config["global_settings"].get("experiment_name"))

        method_options = config["method"]["method_options"]
        result_description = method_options.get("result_description", None)
        input_dir = method_options.get("input_dir", None)

        driver = config.get("driver", None)
        global_settings = config.get("global_settings", None)


        return cls(input_dir, result_description, driver, global_settings)

    def run_simulation(self):
        '''
        Run the FEM simulation through BACI utilities.
        ''' 
        # read and assign some parameters
        current_driver = self.driver
        solver_name = current_driver["driver_params"]["executable_type"].get("solver", None)
        post_processing_name = current_driver["driver_params"]["executable_type"].get("post_processing", None)
        output_prefix = current_driver["driver_params"]["output_prefix"]
        post_paraview = current_driver["driver_params"]["post_paraview"]

        # check if executables exist
        try:
            current_solver = self.global_settings["exe_paths"][solver_name]
        except:
            raise FileNotFoundError(f"Executable {solver_name} does not exist!") 
        
        try:
            current_post_proces = self.global_settings["exe_paths"][post_processing_name]
        except:
            raise FileNotFoundError(f"Executable {post_processing_name} does not exist!") 
        
        # initialize a dictionary to pair dat file and prefix
        dat_prefix_pair = {"dat_file":[],"prefix":[]}

        # iterate through the simulations 
        n_iter = 1
        for inp_file in os.listdir(self.input_dir):
            
            # only if the file is a dat file
            if inp_file.endswith('.dat'):
                output_prefix_current = output_prefix + "_" + str(n_iter)

                # read the solver (baci-release) options    
                exec_options = self.get_solv_options(inp_file, output_prefix_current)
                # call solver 
                self.call_executable(current_solver, exec_options)
                
                simulation_start = f'''
-----------------------------------------------------------------------------------------
**************************** BACI Simulation started *************************************
-----------------------------------------------------------------------------------------
Simulation number: -{n_iter}-
-----------------------------------------------------------------------------------------
                '''
                print(simulation_start)

                # read the post-processing (post_drt_ensight) options 
                exec_options = self.get_post_options(output_prefix_current)
                # call post-processing
                self.call_executable(current_post_proces, exec_options)

                # store the current dat file and its prefix
                dat_prefix_pair["dat_file"].append(inp_file)
                dat_prefix_pair["prefix"].append(output_prefix_current)
                n_iter += 1
        
        if post_paraview:
            self.run_post_paraview(dat_prefix_pair)

    def get_solv_options(self, inp_file, output_prefix_current):
        '''
        Get the solver options from the input file

        Parameters
        ----------
        inp_file : str
            input file within the provided directory
        output_prefix_current : str
            numbered current prefix

        Returns
        -------
        args : list
            arguments/parameters of the called executables 
        
        Notes
        -----
        The usage of baci release as follows
        >>> baci-release input_dir/datfile ouput_dir/output_prefix
        '''
        dat_file = os.path.join(self.input_dir, inp_file)
        output_file = os.path.join(self.global_settings["output_dir"], output_prefix_current)
        args = [dat_file, output_file]
        return args

    def get_post_options(self, output_prefix_current):
        '''
        Get the post processing options from the input file

        Parameters
        ----------
        output_prefix_current : str
            numbered current prefix 
        
        Notes
        -----
        The usage of baci release as follows
        >>> post_drt_ensight input_dir/datfile ouput_dir/output_prefix
        '''

        output_file = os.path.join(self.global_settings["output_dir"], output_prefix_current)
        prefix = '--file='
        args = [prefix + output_file]
        return args


    def call_executable(self, exec_type, args):
        '''
        Calls the chosen executable 

        Parameters
        ----------
        exec_type : str
            executable type to be called
        args : list
            arguments/parameters of the called executables 
        '''

        args.insert(0, exec_type)

        process = subprocess.Popen(args, shell=False, stdout=subprocess.PIPE)
        process.communicate()

        # if printing the BACI output is desired, use the following lines instead of above
        # subprocess.check_output(args)
        # subprocess.call(args) 
    

    def run_post_paraview(self, dat_prefix_pair):
        '''
        Applies post-processing to export Paraview results

        Parameters
        ----------
        dat_prefix_pair : dict
            stores each dat file and its corresponding prefix        
        '''
        try:
            quantity_name = self.result_description["quantity"]["name"]
        except:
            raise ValueError(f"Quantity name in input file not given!")
        try:
            quantity_direction = self.result_description["quantity"]["type"] + "_" + self.result_description["quantity"]["direction"]
        except:
            raise ValueError(f"Quantity direction in input file not given!")
        try:
            level = self.result_description["quantity"]["level"]
        except:
            raise ValueError(f"Quantity level in input file not given!")

        if self.result_description["save_result_in_subfolder"]:
            if not os.path.exists(os.path.join(self.global_settings["output_dir"],"paraview_results")):
                os.makedirs(os.path.join(self.global_settings["output_dir"],"paraview_results"))
            paraview_output_dir = os.path.join(self.global_settings["output_dir"],"paraview_results")
        else:
            paraview_output_dir = self.global_settings["output_dir"]
        
        final_results = {"dat_file":[], quantity_name:[]}

        post_start = f'''
-----------------------------------------------------------------------------------------
*********Post-processing with Paraview started ********************************
-----------------------------------------------------------------------------------------
                '''
        print(post_start)

        # here case files are read randomly, so we have to be sure that paraview results match completely 
        n_case = 0
        for case_file in os.listdir(self.global_settings["output_dir"]):        
            # only if it is a case file
            if case_file.endswith('.case'):
                prefix_case = os.path.splitext(case_file)[0]
                # and only the last 10 letter is as follows
                if prefix_case[-10:] == "_structure":

                    prefix_paraview = prefix_case[:-10]
                    
                    # export the results into a csv file
                    quantity_csv = self.export_paraview_results(case_file, prefix_paraview, paraview_output_dir)

                    quantity_result = pd.read_csv(quantity_csv)

                    if  level == "min":
                        quantity_value = quantity_result.loc[:,quantity_direction].min()
                    elif level == "max":
                        quantity_value = quantity_result.loc[:,quantity_direction].max()
                    else:
                        raise ValueError(f"The given level {level} is not available! Please choose min or max")

                    index_dat = dat_prefix_pair["prefix"].index(prefix_paraview)

                    final_results["dat_file"].append(dat_prefix_pair["dat_file"][index_dat])
                    final_results[quantity_name].append(quantity_value)

                    corr_dat_file = dat_prefix_pair["dat_file"][index_dat]
                    n_case += 1
                    print(f"Post-processing is done for {corr_dat_file}, the total post-processing number: {n_case}")
        
        df_quantity = pd.DataFrame(final_results)
        
        if self.result_description["file_format"] == "csv":
            result_file_name = quantity_name + ".csv"
            df_quantity.to_csv(os.path.join(paraview_output_dir,result_file_name))

    def export_paraview_results(self, case_file, prefix_paraview, paraview_output_dir):
        '''
        Export paraview results into a csv file

        Parameters
        ----------
        case_file : str
            current case file to be investigated
        prefix_paraview : str
            current output prefix for paraview
        paraview_output_dir : str
            path to store paraview results
        
        Returns
        -------
        output_csv : str
            simulation output as a csv file
        '''
        
        #### disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()
        # since Paraview is run on the background for each post-processing and it is not shouted down, memory error arises. To avoid that, reset the session
        self.ResetSession()

        inp_case_file_path = os.path.join(self.global_settings["output_dir"],case_file)
        
        # create a new 'EnSight Reader'
        xxxx_structurecase = simple.EnSightReader(CaseFileName=inp_case_file_path)
        
            # this is taken from Ivo's pvutils (they are more or less same, let's keep it)
            # source = simple.MergeBlocks(Input=xxxx_structurecase)

            # animation_scene = simple.GetAnimationScene()
            # animation_scene.UpdateAnimationUsingDataTimeSteps()
            # animation_scene.GoToLast()

            # view = simple.GetActiveViewOrCreate('RenderView')
            # view.Update()
            # view.ResetCamera()
            # simple.UpdatePipeline()
            # simple.Render()

            # timeKeeper = simple.GetTimeKeeper()
            # timeKeeper.Time = 0.3
            # animation_scene.AnimationTime = 0.3

            # function = ('displacement_2')
            # calc = simple.Calculator(Input=source)
            # calc.Function = function
            # calc.AttributeType = 'Point Data'

            #def get_data_array(input_data):
            #    """
            #    Extract data from an array.
            #    """
            #    data_dir = {}
            #    for i in range(input_data.GetNumberOfArrays()):
            #        data_dir[input_data.GetArrayName(i)] = VN.vtk_to_numpy(
            #            input_data.GetArray(i))
            #    return data_dir

            # data = simple.servermanager.Fetch(calc)

            #point_data = get_data_array(data.GetPointData())
        

        xxxx_structurecase.CellArrays = ['Owner']
        xxxx_structurecase.PointArrays = ['displacement']

        # get animation scene
        animationScene1 = GetAnimationScene()

        # get the time-keeper
        timeKeeper1 = GetTimeKeeper()

        # get active view
        renderView1 = GetActiveViewOrCreate('RenderView') ##commented

        # Create a new 'SpreadSheet View'
        spreadSheetView1 = CreateView('SpreadSheetView')
        spreadSheetView1.ColumnToSort = ''

        # show data in view
        xxxx_structurecaseDisplay_1 = Show(xxxx_structurecase, spreadSheetView1)

        # Properties modified on spreadSheetView1
        spreadSheetView1.GenerateCellConnectivity = 1

        # Properties modified on spreadSheetView1
        spreadSheetView1.HiddenColumnLabels = ['Block Number', 'Point ID', 'Points']

        # Properties modified on animationScene1
        # animationScene1.AnimationTime = 1.0

        # Properties modified on timeKeeper1
        timeKeeper1.Time = 1.0

        # export view
        output_csv = os.path.join(paraview_output_dir, prefix_paraview + '.csv')
        ExportView(output_csv, view=spreadSheetView1)

        #del xxxx_structurecase, animationScene1, timeKeeper1, renderView1, layout1, spreadSheetView1, xxxx_structurecaseDisplay_1
        del xxxx_structurecase, animationScene1, timeKeeper1, spreadSheetView1, xxxx_structurecaseDisplay_1

        return output_csv 
    
    def ResetSession(self):
        '''
        Reset the session for each Paraview to avoid memory allocation.
        '''
        pxm = servermanager.ProxyManager()
        pxm.UnRegisterProxies()
        del pxm
        Disconnect()
        Connect()