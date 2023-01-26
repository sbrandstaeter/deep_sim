"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""
import deepxde as dde
import numpy as np
import tensorflow as tf
import collections
import pandas as pd
from deepxde.metrics import accuracy, l2_relative_error, nanl2_relative_error, mean_l2_relative_error, _absolute_percentage_error, mean_absolute_percentage_error, max_absolute_percentage_error, absolute_percentage_error_std, mean_squared_error


class Pinns():

    def __init__(self, output_dir, result_description, driver, domain, simulation_number, pinn_model_file_name):
        self.output_dir = output_dir
        self.driver = driver
        self.result_description = result_description
        self.domain = domain
        self.simulation_number = simulation_number
        self.pinn_model_file_name = pinn_model_file_name

    @classmethod
    def from_driver_build_model(cls, driver):
        '''
        builds the model from driver 
        '''
        if driver["driver_name"] == "euler_beam":
            from .models.euler_beam import EulerBeam as main_model
        else:
            raise Exception(f"Driver name {driver['driver_name']} does not exists!")
        
        return main_model
    
    def model_output(self, output_file):
        '''
        PINN post-processing.
        
        Parameters
        ----------
        output_file: 
        '''
        
        # initialize the targets and the features 
        targets = collections.OrderedDict()
        features = collections.OrderedDict()
        
        output_options = self.result_description.get("output_options")
        loss_type = output_options.get("loss_type")
        approach = output_options.get("approach")
        metric_name = output_options.get("metric")
        
        file_base = open(output_file,'r')
        
        def initialize_flags():
            train_flag = False
            test_flag = False
            metric_flag = False
            return train_flag, test_flag, metric_flag
        train_flag, test_flag, metric_flag = initialize_flags()
        
        for line in file_base:
            if train_flag:
                train_loss = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            elif test_flag:
                test_loss = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            elif metric_flag:
                metric = np.array([float(i) for i in line.strip().replace("[","").replace("]","").split()])
                train_flag, test_flag, metric_flag = initialize_flags()
            if "Final train loss" in line:
                train_flag = True
            elif "Final test loss" in line:
                test_flag = True
            elif "Final metric" in line:
                metric_flag = True
            
                
        if loss_type == "test":
            final_loss = test_loss
            if approach == "total":
                final_loss = final_loss.sum()
            targets["loss"] = final_loss
        elif loss_type == "train":    
            final_loss = train_loss
            if approach == "total":
                final_loss = final_loss.sum()
            targets["loss"] = final_loss
        
        if metric_name:
            targets["model_accuracy"] = metric.sum()
        
        for feature in self.result_description["features"]:
            features[feature] = self.domain[feature][self.simulation_number]
        
        features.update(targets)
        df = pd.DataFrame.from_dict([features])
        
        for key, _ in targets.items():
            df[key] = df[key].map(lambda x: '%.4e' % x)
            
        file_name = self.output_dir + "/" + self.result_description.get("output_file_name", "default.dat")
        
        if self.simulation_number == 0:
            df.to_csv(file_name, index=False, sep="\t", float_format='%.5f')
        else:
            df.to_csv(file_name, mode="a", index=False, header=False, sep="\t", float_format='%.5f')
    
    def print_targets(self):
        targets = '''
print("Final train loss")
print(losshistory.loss_train[-1])
print("Final test loss")
print(losshistory.loss_test[-1])
print("Final metric")
print(losshistory.metrics_test[-1])
        '''
        return targets