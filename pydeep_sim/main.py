import argparse
import sys
import os
import time
import pathlib
import json
from collections import OrderedDict
import warnings

from pydeep_sim.iterators.iterator import Iterator

def main(args):
    """
    Main function of Deep_Sim

    controls and runs the analysis.

    Args:
        args (list): list of arguments to be parsed
    """
    
    deep_sim_intro = '''
    --------------------------------------------------------------------------------------
    
                *****    ******* *******  *****       *******  ** ***     ***
                **  ***  **      **       **  **      **       ** ****   ****
                **   *** *****   *****    *****       *******  ** ** ** ** **
                **  ***  **      **       **               **  ** **  ***  **
                *****    ******* *******  **          *******  ** **   *   **
    --------------------------------------------------------------------------------------
                         A Multipurpose Python automatization tool
                         to run BEM and FEM simulations and apply
                           Machine and Deep Learning techniques 
    -------------------------------------------------------------------------------------
    '''
    print(deep_sim_intro)

    # read input
    options = get_options(args)

    # build the model
    # breakpoint()
    my_iterator = Iterator.from_config_create_iterator(options)
    
    start_time_calc = time.time()

    print("")
    print("Starting Analysis...")
    print("")

    # perform the simulation
    my_iterator.run()

    end_time_calc = time.time()

    final_time = f'''
---------------------------------------------------------------------------------------
Total Simulation Time: {(end_time_calc - start_time_calc):.3f} seconds
---------------------------------------------------------------------------------------
    '''
    print(final_time)




def get_options(args):
    """
    Parse options from command line and input file.

    Args:
        args (list): list of arguments to be parsed

    Returns:
        dict: parsed options in a dictionary
    """

    parser = argparse.ArgumentParser(description="deep_sim")
    parser.add_argument(
        '--input', type=str, default='input.json', help='Input file in .json format.'
    )
    parser.add_argument('--output_dir', type=str, help='Output directory to write resutls to.')

    args = parser.parse_args(args)

    input_file = os.path.realpath(os.path.expanduser(args.input))
    try:
        with open(input_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise FileNotFoundError("config.json did not load properly.")

    if args.output_dir is None:
        raise Exception("No output directory was given.")

    output_dir = os.path.realpath(os.path.expanduser(args.output_dir))
    if not os.path.isdir(output_dir):
        raise Exception("Output directory does not exist.")

    options["input_file"] = input_file

    # create the "global_settings" dict and move experiment name and output directory into it
    global_settings = {}
    global_settings["output_dir"] = output_dir
    
    try:
        global_settings["experiment_name"] = options["experiment_name"]
    except:
        raise NameError("Experiment name is not defined!")
    
    if "driver" in options:
        global_settings = get_paths(global_settings)
    
    # remove experiment_name field from options dict
    options["global_settings"] = global_settings
    # remove experiment_name field from options dict make copy first
    final_options = dict(options)
    del final_options["experiment_name"]
    return final_options

def get_paths(global_settings):
    '''
    Gets the path for the following executables:
     - baci-release
     - post_drt_ensight
     - bem 

    Parameters
    ----------
    global_settings : dict
        contains output directory and experiment name independent of the given problem

    Returns
    ------
    global_settings : dict
        contains output directory and experiment name independent of the given problem
    '''
    exe_path = {}


    try:
        os.path.isfile(os.environ['BACI_RELEASE'])
        exe_path["baci-release"] = os.environ["BACI_RELEASE"]
    except:
        warnings.warn(("Path to baci-release not found! If you will run baci simulations,"
                        " you will face with errors."
                        "\n"
                        "Be sure that you export BACI_RELEASE env variable and the executable baci-release does exist!\n"))
        
    try:  
        os.path.isfile(os.environ['BACI_POST_DRT_ENSIGHT'])
        exe_path["post_drt_ensight"] = os.environ["BACI_POST_DRT_ENSIGHT"]
    except:
        warnings.warn(("Path to post_drt_ensight not found! If you will post process the BACI Simulations,"
                        " you will face with errors."
                        "\n"
                        "Be sure that you export BACI_POST_DRT_ENSIGHT env variable and the executable post_drt_ensight does exist!\n"))

    try: 
        os.path.isfile(os.environ['BEM'])
        exe_path["bem"] = os.environ["BEM"]
    except:
        warnings.warn(("Path to bem not found! If you will run BEM simulations,"
                        " you will face with errors."
                        "\n"
                        "Be sure that you export BEM env variable and the executable bem does exist!\n"))
        
    try: 
        os.path.isfile(os.environ['MIRCO'])
        exe_path["mirco"] = os.environ["MIRCO"]
    except:
        warnings.warn(("Path to MIRCO not found! If you will run rough surface contact simulations vi MIRCO,"
                        " you will face with errors."
                        "\n"
                        "Be sure that you export BEM env variable and the executable bem does exist!\n"))
    
    global_settings["exe_paths"] = exe_path
    
    return global_settings

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))