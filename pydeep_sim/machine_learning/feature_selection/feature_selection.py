class FeatureSelection:
    def __init__(self, X_train, X_test, y_train, y_test, feature_selection_block):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_selection_block = feature_selection_block


    def build_filter(self):

        '''
        builds the filtering method to select the features

        some options for different filters

        - select_k_best:
                        - score_func:
                            - f_classif    --> linear --> categorical output
                            - f_regression --> same with pearson
                            - mutual_info_regression  --> mix model
                        - k
        
        - variance_threshold:
                            - threshold

        - correlation_matrix:
                            - method:
                                - pearson  --> linear    --> numerical output
                                - kendall  --> nonlinear --> categorical output
                                - spearman --> nonlinear --> numerical output
                            - k
        '''  

        from sklearn.feature_selection import SelectKBest 
        from sklearn.feature_selection import VarianceThreshold

        filter_dict = {
            "select_k_best" : SelectKBest(),
            "variance_threshold" : VarianceThreshold(),
            "correlation_matrix" : "correlation_matrix"
        }

        try:
            current_filter = filter_dict[self.feature_selection_block["filtering"]["type"]]
        except:
            raise NameError("The chosen feature selection method is not available!")

        return current_filter
    
    
    def build_wrapper(self):

        from sklearn.feature_selection import SelectFromModel
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.feature_selection import RFE
        from sklearn.feature_selection import RFECV

        # regresssion models
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import Ridge
        from sklearn.linear_model import Lasso
        from sklearn.linear_model import ElasticNet
        from sklearn.linear_model import BayesianRidge
        from sklearn.linear_model import PassiveAggressiveRegressor
        from sklearn.svm import LinearSVR
        from sklearn.svm import SVR
        from sklearn.linear_model import SGDRegressor
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor

        # classification models
        from sklearn.linear_model import RidgeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import SGDClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier


        estimator_dict = {
            # regression models
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
            
            # classification models
            "ridge_classification" : RidgeClassifier(),
            "logistic_classification" : LogisticRegression(),
            "pas_agg_classification" : PassiveAggressiveClassifier(),
            "svm_lin_classification" : LinearSVC(),
            "svm_classification" : SVC(),
            "sgd_classification" : SGDClassifier(),
            "kneigh_classification" : KNeighborsClassifier(),
            "decision_tree_classification" : DecisionTreeClassifier(),
            "random_forest_classification" : RandomForestClassifier(),
        }


        # build the estimator with options if it exists
        try:
            estimator_type = self.feature_selection_block["wrapper"]["estimator"]["type"]
            estimator_options = self.feature_selection_block["wrapper"]["estimator"]["options"]
            estimator = estimator_dict[estimator_type].set_params(**estimator_options)
        except:
            raise NameError("The chosen estimator type is not available!")

        wrapper_dict = {
            "select_from_model" : SelectFromModel(estimator),
            "sequential_feature_selector" : SequentialFeatureSelector(estimator),
            "recursive_feature_elimination" : RFE(estimator),
            "recursive_feature_elimination_with_CV" : RFECV(estimator),
        }

        # build th wrapper with options check if it exists
        try:
            wrapper_type = self.feature_selection_block["wrapper"]["type"]
            wrapper_options = self.feature_selection_block["wrapper"]["options"]
            wrapper = wrapper_dict[wrapper_type].set_params(**wrapper_options)
        except:
            raise NameError("The chosen wrapper type is not available!")
        
        return wrapper


    def generate_filter(self):
        
        current_filter = self.build_filter()

        if current_filter == "correlation_matrix":

            if self.y_train.shape[1] != 1:
                raise ValueError("Correlation matrix can not be applied if more than one target is available!" + 
                                " Choose a different feature selection method!")

            method = self.feature_selection_block["filtering"]["options"]["method"]
            correlation = self.X_train.corrwith(self.y_train.iloc[:,0], method=method)
            limit = self.feature_selection_block["filtering"]["options"]["k"]

            selected_features = correlation.abs().sort_values(ascending=False)[:limit].index

            return selected_features

        else:

            filter_args = self.feature_selection_block["filtering"]["options"]

            if "score_func" in filter_args:

                from sklearn.feature_selection import f_classif, f_regression, mutual_info_regression

                score_func_dic = {
                "f_classif" : f_classif,
                "f_regression" : f_regression,
                "mutual_info_regression" : mutual_info_regression
                }

                filter_args["score_func"] = score_func_dic[filter_args["score_func"]]

            current_filter.set_params(**filter_args)

            # fit train set
            current_filter.fit(self.X_train,self.y_train.values.ravel())

            mask = current_filter.get_support()

            selected_features = [] # The list of your K best features
            feature_names = self.X_train.columns

            for cond, feature in zip(mask, feature_names):
                if cond:
                    selected_features.append(feature)
                        

            return selected_features

    def generate_wrapper(self):
        wrapper = self.build_wrapper()

        wrapper.fit(self.X_train,self.y_train.values.ravel())

        mask = wrapper.get_support()

        selected_features = [] # The list of your K best features
        feature_names = self.X_train.columns

        for cond, feature in zip(mask, feature_names):
            if cond:
                selected_features.append(feature)
                    
        return selected_features

    def perform_feature_selection(self):

        if self.feature_selection_block["filtering"]:
            selected_features = self.generate_filter()
            self.X_train = self.X_train[selected_features]
            self.X_test = self.X_test[selected_features]

        if self.feature_selection_block["wrapper"]:
            selected_features = self.generate_wrapper()   
            self.X_train = self.X_train[selected_features]
            self.X_test = self.X_test[selected_features] 
        
        return self.X_train, self.X_test

        