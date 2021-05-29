import os
import pytest
import json

from pimcsml.main import main


def test_baci_fem_iterator(generate_json,tmpdir):
    print(generate_json)
    arguments = [
        '--input=' + generate_json,
        '--output=' + str(tmpdir),
    ]
#    breakpoint()
    main(arguments)

    os.remove(generate_json)
    
    pass


@pytest.fixture
def generate_json():
    """ Return the path to the json input-files of the function test. """
    dirpath = os.path.dirname(__file__)
    input_dir = os.path.join(dirpath, 'input_files/baci_inputs')
    json_input_file = os.path.join(dirpath, 'input_files')

    data = {
                "experiment_name" : "Rough Surface Contact Run Automatized Fem Simulation",
                "method": {
                    "method_name": "baci_fem",
                    "method_options" :{
                        "result_description" : {
                            "write_results" : True
                        },
                        "input_dir": input_dir
                    }
                },
                "driver" :{
                    "driver_type"           : "baci_driver",
                    "driver_params"         :{
                        "executable_type"    : {
                            "solver" : "baci-release",
                            "post_processing" : "post_drt_ensight"
                        },
                        "output_prefix" : "baci_test_run_example"
                    }
                }
            }
    
    file_name = os.path.join(json_input_file, 'input_rsc_fem_iterator.json')
    
    with open(file_name, 'w') as out_file:
        json.dump(data, out_file, indent=4)

    return file_name