import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections


from .iterator import Iterator
from ..ml_preprocess.ml_preprocess import MachineLearningPreprocess

class MachineLearningIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, result_description, model_input_file, model, ml_preprocess, global_settings):
        super(MachineLearningIterator, self).__init__(None, global_settings)
        self.result_description = result_description
        self.model_input_file = model_input_file
        self.model = model
        self.ml_preprocess = ml_preprocess
    
    @classmethod
    def from_config_create_iterator(cls, config, iterator_name=None):

        print(config.get("global_settings", None)["experiment_name"])

        method_options = config["method"].get("method_options")
        result_description = method_options.get("result_description")
        
        try:
            model = config["model"]
        except:
            raise NameError("The model is not defined in JSON input!")     
        
        try:
            model_input_file = config["model_input_file"]
        except:
            raise NameError("The input file is not provided in JSON input!")   

        try:
            ml_preprocess = config["ml_preprocess"]
        except:
            raise NameError("Preprocessing is not defined in JSON input!")

        global_settings = config.get("global_settings", None)

        return cls(result_description, model_input_file, model, ml_preprocess, global_settings)

    def run_simulation(self):
        '''
        Perform the Machine Learning 
        '''
        # load the data       
        data = self.load_data()

        # perform the preprocessing
        preprocess_block = self.ml_preprocess
        ml_pre_process = MachineLearningPreprocess(data=data, preprocess_block=preprocess_block)
        X_train, X_test, y_train, y_test = ml_pre_process.generate_processed_data()
        

    def load_data(self):

        m_input_file = self.model_input_file 
        
        if m_input_file.get("read_options"):
            path_name = {"filepath_or_buffer":m_input_file["model_input_file_name"]}
            args_from_input = m_input_file["read_options"].get("reader_options")
            args = {**path_name, **args_from_input}
            
            data = getattr(pd, m_input_file["read_options"]["reader"])(**args)    
        else:
            # the default is read_csv
            data = getattr(pd, 'read_csv')()

        return data