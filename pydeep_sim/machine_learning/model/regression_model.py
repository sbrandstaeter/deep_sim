import pandas as pd
import time

from sklearn.base import BaseEstimator

class RegressionModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_block, global_settings):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_block = model_block
        self.global_settings = global_settings

    def build_model(self):

        # regresssion models
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import BayesianRidge
        from sklearn.linear_model import PassiveAggressiveRegressor
        from sklearn.linear_model import SGDRegressor
        from sklearn.svm import LinearSVR
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.ensemble import AdaBoostRegressor
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.gaussian_process import GaussianProcessRegressor
        from xgboost import XGBRegressor
        
        regr_dict = {
            "linear_regression" : LinearRegression(),
            "ridge_regression" : Ridge(),
            "lasso_regression" : Lasso(),
            "elastic_net" : ElasticNet(),
            "bayesian_ridge" : BayesianRidge(),
            "pas_agg_regression" : PassiveAggressiveRegressor(),
            "svm_lin_regression" : LinearSVR(),
            "svm_regression" : SVR(),
            "sgd_regression" : SGDRegressor(),
            "kneigh_regression" : KNeighborsRegressor(),
            "decision_tree_regression" : DecisionTreeRegressor(),
            "random_forest_regression" : RandomForestRegressor(),
            "adaboost_regression" : AdaBoostRegressor(),
            "grad_boost_regression" : GradientBoostingRegressor(),
            "kernel_ridge" : KernelRidge(),
            "gaussian_regression":GaussianProcessRegressor,
            "xgboost_regression" : XGBRegressor()
        }

        try:
            model = regr_dict[self.model_block["model_type"]]
        except:
            raise NameError("The chosen model type is not available!")
        
        return model

    
    def build_tuner(self):

        # tuning or hyperparameter optimization libraries
        from sklearn.model_selection import GridSearchCV
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingGridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import HalvingRandomSearchCV

        tuner_dict = {
            "grid_search_cv" : GridSearchCV,
            "halving_grid_search_cv" : HalvingGridSearchCV,
            "random_search_cv" : RandomizedSearchCV,
            "halving_random_search_cv" : HalvingRandomSearchCV
        }

        try:
            tuner = tuner_dict[self.model_block["parameter_tuning"]["type"]]
        except:
            raise NameError("The chosen parameter tuning is not available!")
        
        return tuner

    def perform_train_test(self):
    
        # define the trainer
        trainer = self.set_options()

        # build metrics
        if self.model_block["evaluation_metrics"]:
            eval_metrics = self.build_evaluation(self.model_block["evaluation_metrics"])
        else:
            eval_metrics = None

        # if hyperparameter optimization is desired, then
        # find the model with best parameter (best estimator)
        if self.model_block["parameter_tuning"]:
            trainer = self.find_best_model(trainer) # this is already a fitted model as long as refit=True
            
        # Fit the model to measure time and if hyperparameter optimization is not done
        start_time_fit = time.time()
        trainer.fit(self.X_train, self.y_train.values.ravel())
        end_time_fit = time.time()

        print(f"Fit time: {(end_time_fit - start_time_fit):.3f} seconds. \n")

        # evaluate the train and test error
        start_time_pred = time.time()
        y_pred_train = trainer.predict(self.X_train)
        end_time_pred = time.time()
        y_pred_test = trainer.predict(self.X_test)

        print(f"Prediction time: {(end_time_pred - start_time_pred):.3f} seconds. \n")

        if eval_metrics:
            for key, value in eval_metrics.items():
                print(f"The training error for {key} is : {(value(y_pred_train,self.y_train)):.5f}.")
                print(f"The test error for {key} is     : {(value(y_pred_test,self.y_test)):.5f}. \n")


    def set_options(self):
        # check if hyperparameter optimization is desired
        if self.model_block["parameter_tuning"]:
            
            # build tuner and model
            tuner = self.build_tuner()  
            model = self.build_model()
            
            # set the tuning parameters
            tuner_options = self.model_block["parameter_tuning"]["options"]
            tuner_options["estimator"] = model
            tuner = tuner(**tuner_options)   
            
            return tuner
        
        else:
            # build the model
            model = self.build_model()
            # set the model parameters if available
            if self.model_block.get("model_options"):
                model_parameters = self.model_block["model_options"]
                model.set_params(**model_parameters)
            
            return model


    def find_best_model(self, trainer):

        start_time_calc = time.time()

        print('''
*************************************************************************
--------------- Hyperparameter optimization is running! -----------------
*************************************************************************       
        ''')
        
        trainer.fit(self.X_train, self.y_train.values.ravel())

        end_time_calc = time.time()

        print(f'''
---------------------------------------------------------------------------------------
Total Hyperparameter optimization time: {(end_time_calc - start_time_calc):.3f} seconds
---------------------------------------------------------------------------------------\n
        ''')

        print(f"The best model is: {trainer.best_estimator_} \n")

        # store all of the models in hyper_parameter_models.dat file
        file_name = "/hyper_parameter_models"
        extension = ".dat"
        hyper_file = self.global_settings["output_dir"] + file_name + extension

        df_hyper_models = pd.DataFrame(trainer.cv_results_)
        
        df_hyper_models.to_csv(hyper_file, sep="\t")

        return trainer.best_estimator_

    def build_evaluation(self, eval_metrics):
        
        # import metrics
        
        from sklearn.metrics import mean_absolute_error, max_error
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import mean_squared_error, mean_squared_log_error
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance

        metrics_dict = {
            "mean_absolute_error" : mean_absolute_error,
            "explained_variance_score" : explained_variance_score,
            "mean_squared_error" : mean_squared_error,
            "mean_squared_log_error" : mean_squared_log_error,
            "r2_score" : r2_score,
            "mean_poisson_deviance" : mean_poisson_deviance,
            "mean_gamma_deviance" :mean_gamma_deviance,
            "max_error" : max_error
        }

        metrics = {}
        for metric in eval_metrics:
            try:
                metrics[metric] = metrics_dict[metric]
            except:
                raise NameError("The chosen evaluation type (metric) is not available!")

        return metrics



            
