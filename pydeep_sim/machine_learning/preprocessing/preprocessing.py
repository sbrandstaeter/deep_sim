# libraries for imputing the missing values
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# libraries for scaling the data
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

# generates the polynomial features
from sklearn.preprocessing import PolynomialFeatures

# categorical encoders
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer

import pandas as pd

from sklearn.model_selection import train_test_split


class MachineLearningPreprocess:
    
    def __init__(self, data, preprocess_block):
        self.data = data
        self.preprocess_block = preprocess_block

    def build_imputing(self):

        imputer_dict = {
            "simple_imputer" : SimpleImputer(),
            "iterative_imputer" : IterativeImputer(),
            "knn_imputer" : KNNImputer()
        }

        try:
            if self.preprocess_block["imputer"]["options"]:
                imp_args = self.preprocess_block["imputer"]["options"]
                ml_imputer = imputer_dict[self.preprocess_block["imputer"]["type"]].set_params(**imp_args)
            else:
                ml_imputer = imputer_dict[self.preprocess_block["imputer"]["type"]]
        except:
            raise NameError("The chosen imputing method is not available!")

        return ml_imputer


    def build_encoding(self):
        
        encoder_dict = {
            "binarizer" : Binarizer(),
            "label_binarizer" : LabelBinarizer(),
            "label_encoder" : LabelEncoder(),
            "one_hot_encoder" : OneHotEncoder()
        }

        try:
            if self.preprocess_block["encoder"]["options"]:
                encod_args = self.preprocess_block["encoder"]["options"]
                ml_encoder = encoder_dict[self.preprocess_block["encoder"]["type"]].set_params(**encod_args)
            else:
                ml_encoder = encoder_dict[self.preprocess_block["encoder"]["type"]]
        except:
            raise NameError("The chosen encoding method is not available!")

        return ml_encoder


    def build_scaling(self):
    
        scaler_dict = {
            "normalizer" : Normalizer(),
            "standard_scaler" : StandardScaler(),
            "min_max_scaler" : MinMaxScaler(),
            "max_abs_scaler" : MaxAbsScaler()
        }

        try:
            scaler_args = self.preprocess_block["scaler"]["options"]
            ml_scaler = scaler_dict[self.preprocess_block["scaler"]["type"]].set_params(**scaler_args)
        except:
            raise NameError("The chosen scaling method is not available!")

        return ml_scaler

        
    def drop_entities(self):
        
        dropped_features = self.preprocess_block["dropper"]

        if set(dropped_features).issubset(set(self.data.columns)):
            self.data.drop(dropped_features, axis=1, inplace= True)
        else:
            raise ValueError("Feautures to drop do not exist in data!")

        return self.data
    
    def generate_processed_data(self):
        
        # drop the entities from the data
        self.drop_entities()

        # chooose targets and features
        targets = self.data[self.preprocess_block["targets"]]
        features = self.data.drop(targets, axis=1)

        # split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, **(self.preprocess_block["model_split"]))

        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)     

        # apply encoding on the desired columns and check if really the data type of those columns is an object, 
        # otherwise it would not make sense
        if self.preprocess_block.get("encoder"):
            
            enc = self.build_encoding()

            encoded_columns = self.preprocess_block["encoder"]["feature_names"]

            for feature in encoded_columns:
                if (X_train[feature].dtypes == 'object'):
                    # perform encoding
                    enc.fit(X_train[feature])
                    # transform on train
                    X_train_transformed = enc.transform(X_train[feature])
                    # transform on test
                    X_test_transformed = enc.transfrom(X_test[feature])

                    dim = X_train_transformed.shape[1]
                    new_name = [feature + "_encoded_" + str(i+1) for i in range(dim)]

                    X_train_tr_df = pd.DataFrame(X_train_transformed, columns=new_name)
                    X_test_tr_df = pd.DataFrame(X_test_transformed, columns=new_name)

                    X_train = pd.concat([X_train, X_train_tr_df], axis=1)
                    X_test = pd.concat([X_test, X_test_tr_df], axis=1)

                    X_train.drop(feature, axis=1, inplace=True)
                    X_test.drop(feature, axis=1, inplace= True)
    
        if self.preprocess_block.get("imputer"):

            if X_train.isnull().any().any() or y_train.isnull().any().any():

                imp = self.build_imputing()
                # ----------------------
                # imput the X (features)
                # ----------------------
                X_column_names = X_train.columns # obtain the column names
                #perform imputing
                imp.fit(X_train)
                # transform on train
                X_train_transformed = imp.transform(X_train)
                # transform on test
                X_test_transformed = imp.transform(X_test)

                # convert transformed array into dataframes
                X_train = pd.DataFrame(X_train_transformed, columns=X_column_names)
                X_test = pd.DataFrame(X_test_transformed, columns=X_column_names)

                # ------------------
                # imput the y (targets)
                # ------------------

                y_column_names = y_train.columns # obtain the column names
                #perform imputing
                imp.fit(y_train)
                # transform on train
                y_train_transformed = imp.transform(y_train)
                # transform on test
                y_test_transformed = imp.transform(y_test)

                # convert transformed array into dataframes
                y_train = pd.DataFrame(y_train_transformed, columns=y_column_names)
                y_test = pd.DataFrame(y_test_transformed, columns=y_column_names)

        if self.preprocess_block.get("scaler"):
            
            sc = self.build_scaling()

            X_column_names = X_train.columns # obtain the column names
            #perform imputing
            sc.fit(X_train)
            # transform on train
            X_train_transformed = sc.transform(X_train)
            # transform on test
            X_test_transformed = sc.transform(X_test)

            # convert transformed array into dataframes
            X_train = pd.DataFrame(X_train_transformed, columns=X_column_names)
            X_test = pd.DataFrame(X_test_transformed, columns=X_column_names)

        return X_train, X_test, y_train, y_test
             

        