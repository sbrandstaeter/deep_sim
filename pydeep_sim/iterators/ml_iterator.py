import numpy as np
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections

from sklearn import feature_selection


from .iterator import Iterator
from ..machine_learning.preprocessing.preprocessing import MachineLearningPreprocess
from ..machine_learning.preprocessing.polynomial_features import DeepPolynomialFeatures
from ..machine_learning.feature_selection.feature_selection import FeatureSelection
from ..machine_learning.model.regression_model import RegressionModel
from ..machine_learning.model.classification_model import ClassificationModel

class MachineLearningIterator(Iterator):
    '''
    This module generates rough surfaces and runs BEM simulations
    ''' 
    
    def __init__(self, result_description, model_input_file, model, preprocessing, feature_selection, polynomial_features, global_settings):
        super(MachineLearningIterator, self).__init__(None, global_settings)
        self.result_description = result_description
        self.model_input_file = model_input_file
        self.model = model
        self.preprocessing = preprocessing
        self.feature_selection = feature_selection
        self.polynomial_features = polynomial_features
        self.model = model
    
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
            preprocessing = config["preprocessing"]
        except:
            raise NameError("Preprocessing is not defined in JSON input!")

        feature_selection = config.get("feature_selection")
        polynomial_features = config.get("polynomial_features")

        global_settings = config.get("global_settings", None)

        return cls(result_description, model_input_file, model, preprocessing, feature_selection, polynomial_features, global_settings)

    def run_simulation(self):
        '''
        Perform the Machine Learning 
        '''
        # load the data       
        data = self.load_data()

        # perform the preprocessing
        preprocess_block = self.preprocessing
        preprocesser = MachineLearningPreprocess(data=data, preprocess_block=preprocess_block)
        X_train, X_test, y_train, y_test = preprocesser.generate_processed_data()

        # perform feature selection if available
        if self.feature_selection:
            feature_selection_block = self.feature_selection
            feature_selecter = FeatureSelection(X_train=X_train, X_test= X_test, y_train=y_train, y_test=y_test, feature_selection_block=feature_selection_block)
            X_train, X_test = feature_selecter.perform_feature_selection()
        
        # perform polynomial features if available
        if self.polynomial_features:
            polynomial_features_block = self.polynomial_features
            polynomial_features_selecter = DeepPolynomialFeatures(X_train=X_train, X_test= X_test, y_train=y_train, polynomial_features_block=polynomial_features_block)
            X_train, X_test = polynomial_features_selecter.perform_poly_features()

        # perform the training, and test the model
        model_block = self.model
        if model_block["problem_type"] == "regression":
            regr_model = RegressionModel(X_train=X_train, X_test= X_test, y_train=y_train, y_test=y_test, model_block=model_block, global_settings=self.global_settings)
            regr_model.perform_train_test()
        elif model_block["problem_type"] == "classification":
            clas_model = ClassificationModel(X_train=X_train, X_test= X_test, y_train=y_train, y_test=y_test, model_block=model_block)
            clas_model.perform_train_test()
        else:
            raise NameError("The given model type is not available!")

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