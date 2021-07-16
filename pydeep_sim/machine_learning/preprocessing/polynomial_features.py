# generates the polynomial features
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class DeepPolynomialFeatures:

    def __init__(self, X_train, X_test, y_train, polynomial_features_block):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.polynomial_features_block = polynomial_features_block

    
    def build_poly_features(self):
    
        # build the polynomial features
        poly_features_dict = {
            "poly_features" : PolynomialFeatures()
        }

        try:
            poly_features = poly_features_dict[self.polynomial_features_block["type"]]
        except:
            raise NameError("The chosen polynomial features method is not available!")

        
        if self.polynomial_features_block.get("options"):
            poly_feat_params = self.polynomial_features_block["options"]
            poly_features.set_params(**poly_feat_params)

        return poly_features
    
    def perform_poly_features(self):  
    
        pf = self.build_poly_features()

        if self.polynomial_features_block.get("choose"):
            # apply poly_features only on the chosen columns
            chosen_cloumns = self.polynomial_features_block["choose"]
            X_train_chosen = self.X_train[chosen_cloumns]
            X_test_chosen = self.X_test[chosen_cloumns]
            
            # fit the model 
            pf.fit(X_train_chosen)
            # transform on train
            X_train_chosen = pf.transform(X_train_chosen)
            # transform on test
            X_test_chosen = pf.transform(X_test_chosen)

            self.X_train.drop(chosen_cloumns, axis=1, inplace=True)
            self.X_test.drop(chosen_cloumns, axis=1, inplace=True)

            self.X_train = pd.concat([self.X_train, pd.DataFrame(X_train_chosen)], axis=1)
            self.X_test = pd.concat([self.X_test, pd.DataFrame(X_test_chosen)], axis=1)

        else:
            # fit the model 
            pf.fit(self.X_train)
            # transform on train
            self.X_test = pf.transform(self.X_test)
            # transform on test
            self.X_test = pf.transform(self.X_test)

        return self.X_train, self.X_test