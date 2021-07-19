

class ClassificationModel:

    def __init__(self, X_train, X_test, y_train, y_test, model_block):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_block = model_block

    def build_model(self):
        
        # hyperparameter optimization libraries
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import HalvingGridSearchCV
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import HalvingRandomSearchCV

        # classification models
        from sklearn.linear_model import RidgeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.linear_model import PassiveAggressiveClassifier
        from sklearn.linear_model import SGDClassifier
        from sklearn.svm import SVC
        from sklearn.svm import LinearSVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.ensemble import GradientBoostingClassifier
        from xgboost import XGBClassifier

    


    