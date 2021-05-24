import argparse
import sys
import os
import time
import pathlib
import json
from collections import OrderedDict

from pimcsml.iterators.iterator import Iterator

def main(args):
    """
    Main function of imcsML

    controls and runs the analysis.

    Args:
        args (list): list of arguments to be parsed
    """
    print("hello world")
    
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
    print("")
    print(f"Time for CALCULATION: {end_time_calc - start_time_calc} s")
    print("")




def get_options(args):
    """
    Parse options from command line and input file.

    Args:
        args (list): list of arguments to be parsed

    Returns:
        dict: parsed options in a dictionary
    """

    parser = argparse.ArgumentParser(description="IMCSML")
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

    # move some parameters into a global settings dict to be passed to e.g.
    # iterators facilitating input output stuff
    global_settings = {}
    global_settings["output_dir"] = output_dir
    global_settings["experiment_name"] = options["experiment_name"]
    
    if "driver" in options:
        try:
            executable_path = str(pathlib.Path(__file__).parents[1].joinpath("executables"))
            global_settings["executable_path"] = executable_path
        except:
            raise Exception("Executables folder does not exist!")

    # remove experiment_name field from options dict
    options["global_settings"] = global_settings
    # remove experiment_name field from options dict make copy first
    final_options = dict(options)
    del final_options["experiment_name"]
    return final_options

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))