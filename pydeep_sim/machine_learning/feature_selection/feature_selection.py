from sklearn.feature_selection import SelectKBest 

from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression


class FeatureSelection:
    def __init__(self, X_train, y_train, feature_selection_block):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_selection_block = feature_selection_block


    def build_feature_selecter(self):

        imputer_dict = {
            "select_k_best" : SelectKBest(),
        }

        try:
            if self.feature_selection_block["options"]:
                select_args = self.feature_selection_block["options"]
                feature_selecter = imputer_dict[self.feature_selection_block["type"]].set_params(**select_args)
            else:
                feature_selecter = imputer_dict[self.feature_selection_block["type"]]
        except:
            raise NameError("The chosen feature selection method is not available!")

        return feature_selecter
    
    
    def select_features(self):
        

        pass
             

        