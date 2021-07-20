import os
import pytest
import json

from pydeep_sim.main import main


def test_ml_iterator(generate_json,tmpdir):
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
    input_file = os.path.join(dirpath, 'input_files/ml_inputs/simulation_output_Rough_Surface_Contact_Automatized_Simulation.dat')
    json_input_file = os.path.join(dirpath, 'input_files')


    data = {
            "experiment_name" : "Machine Learning model generator and trainer",
            "method":   {
                        "method_name": "ml_trainer",
                        "method_options" :  {     
                                            "result_description" :  {
                                                                    "write_results" : True
                                                                    },
                                            }
                        },
            
            "model_input_file" : {
                            "model_input_file_name" : input_file,
                            "read_options" : {
                                                "reader" : "read_csv",
                                                "reader_options":   {
                                                                    "sep" : "\t" 
                                                                    }
                                                    }
                            },

            "preprocessing" :   {
                                "targets" : ["total_cont_area"],
                                "dropper" : ["E1", "nu1", "E2", "nu2", "lato", "g0", "Delta", "errf", "tol", "total_force"],
                                "model_split":  {
                                                "train_size" : 0.8
                                                },
                                "encoder" : {
                                            "type":"label_binarizer",
                                            "feature_names" : ["n"],           
                                            "options" : {
                                                        "sparse_output" : False
                                                        }           
                                            },
                                "imputer" : {
                                            "type":"simple_imputer",           
                                            "options" : {
                                                        "strategy" : "mean"
                                                        }           
                                            },               
                                "scaler" : {
                                            "type":"normalizer",           
                                            "options" : {
                                                        "norm" : "l2"
                                                        }           
                                            },
                                },

            "polynomial_features1" : {
                                    "type": "poly_features",
                                    "options":  {
                                                "degree" : 2
                                                },
                                    "choose" : ["n","alfa_y"] 
                                    },

            "feature_selection" :   {
                                    "filtering" :  {
                                                "type": "select_k_best",
                                                "options" : {
                                                            "score_func" : "f_regression",
                                                            "k" : 5
                                                            }
                                                },
                                    "wrapper1" : {
                                                "type" : "select_from_model",
                                                "options" : {
                                                            "max_features" : 20
                                                            },
                                                "estimator" :   {
                                                                "type" : "svm_lin_regression",
                                                                "options" : {
                                                                            "C" : 0.5
                                                                            }
                                                                }
                                                }
                                     },

            "model" :   {
                        "problem_type" : "regression",
                        "multi_targets" : False,
                        "model_type" : "decision_tree",
                        "model_options":    {

                                            },
                        "parameter_tuning": { 
                                            "type" : "grid_search_cv",
                                            "options":  {
                                                        "param_grid" :  {
                                                                        'min_samples_leaf': [0.1, 0.3, 0.5],
                                                                        'min_samples_split': [0.1, 0.3, 0.5, 0.7, 1.0],
                                                                        'max_depth': [1, 5, 15, 30, None]},
                                                        "n_jobs" : -1
                                                        }
                                            },
                        "evaluation_metrics" : ["mean_squared_error","mean_absolute_error"],
                        "save_model" :  {
                                        "name" : "decision_tree_base",
                                        "extension" : ".pkl"
                                        }
                        }
            }
    
    file_name = os.path.join(json_input_file, 'input_ml_iterator_from_test.json')
    
    with open(file_name, 'w') as out_file:
        json.dump(data, out_file, indent=4)

    return file_name